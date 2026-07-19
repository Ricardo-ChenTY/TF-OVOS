#!/usr/bin/env python
"""Real, bounded Observation-5 diagnostic pilot for CorrCLIP.

This is deliberately separate from ``generate_diagnostic_tables.py``.  The
latter reconstructs diagnostics from CorrCLIP's saved dense argmax maps;
this script makes new model calls on a fixed subset of Context-459 and
ADE-847, and creates SAM proposals afresh.

The four measurements are image/class (semantic-region) averages.  A region
is the union of all GT pixels bearing one valid semantic class ID in an image.
For localization, CorrCLIP's correlation-refined feature map is computed once
per image.  Each GT class is then scored using *only* its own prompt embedding;
Otsu thresholding of that single score map creates the binary prediction.  A
one-class softmax would label every pixel foreground, so it is intentionally
not used.

Typical invocation (the project environment is required):

  /data/tianyi/conda_envs/tf-ovos/bin/python scripts/real_obs5_diagnostic_pilot.py

The default is 250 images per dataset.  ``--num-images 1`` is useful as a
smoke test, and ``--datasets context459`` can be used for a bounded rerun.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CORRCLIP_ROOT = ROOT / "third_party" / "official_methods" / "CorrCLIP"
DEFAULT_OUT_CSV = ROOT / "runs" / "analysis" / "obs5_real_diagnostic_pilot.csv"
DEFAULT_OUT_MD = ROOT / "runs" / "analysis" / "obs5_real_diagnostic_pilot.md"
SAM_CHECKPOINT = ROOT / "weights" / "sam_vit_h_4b8939.pth"
METACLIP_CHECKPOINT = Path.home() / ".cache" / "clip" / "b16_fullcc2.5b.pt"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    manifest: Path
    vocab: Path
    corrclip_names: Path
    corrclip_masks: Path
    num_classes: int


SPECS = {
    "context459": DatasetSpec(
        key="context459",
        display_name="Context-459",
        manifest=ROOT / "data" / "manifests" / "context459_val.jsonl",
        vocab=ROOT / "configs" / "vocab" / "context_459.txt",
        corrclip_names=CORRCLIP_ROOT / "configs" / "cls_context459.txt",
        corrclip_masks=CORRCLIP_ROOT / "data" / "region_masks" / "context",
        num_classes=459,
    ),
    "ade847": DatasetSpec(
        key="ade847",
        display_name="ADE-847",
        manifest=ROOT / "data" / "manifests" / "ade20k847_val.jsonl",
        vocab=ROOT / "configs" / "vocab" / "ade20k_847.txt",
        corrclip_names=CORRCLIP_ROOT / "configs" / "cls_ade20k847.txt",
        corrclip_masks=CORRCLIP_ROOT / "data" / "region_masks" / "ade",
        num_classes=847,
    ),
}


@dataclass
class DiagnosticAccumulator:
    spec: DatasetSpec
    requested_images: int
    attempted_images: int = 0
    processed_images: int = 0
    naming_regions: int = 0
    naming_hits: int = 0
    localization_regions: int = 0
    localization_iou_sum: float = 0.0
    proposal_regions: int = 0
    proposal_iou_sum: float = 0.0
    proposal_recall_hits: int = 0
    errors: Counter[str] = field(default_factory=Counter)
    error_examples: list[str] = field(default_factory=list)
    corrclip_missing_mask_fallbacks: int = 0
    sam_empty_images: int = 0
    sam_proposals_total: int = 0

    def note_error(self, stage: str, sample_id: str, exc: BaseException) -> None:
        key = f"{stage}:{type(exc).__name__}"
        self.errors[key] += 1
        if len(self.error_examples) < 8:
            self.error_examples.append(f"{sample_id} [{stage}]: {type(exc).__name__}: {exc}")

    def row(self, backbone: str, prompt_method: str, localization_threshold: str) -> dict[str, Any]:
        denom_naming = max(self.naming_regions, 1)
        denom_loc = max(self.localization_regions, 1)
        denom_prop = max(self.proposal_regions, 1)
        return {
            "row_type": "dataset",
            "dataset": self.spec.display_name,
            "dataset_key": self.spec.key,
            "method": "CorrCLIP",
            "diagnostic_source": "real_model_calls_pilot",
            "gt_region_naming_top1": self.naming_hits / denom_naming,
            "gt_text_localization_iou": self.localization_iou_sum / denom_loc,
            "proposal_oracle_iou": self.proposal_iou_sum / denom_prop,
            "proposal_recall_at_05": self.proposal_recall_hits / denom_prop,
            "requested_images": self.requested_images,
            "attempted_images": self.attempted_images,
            "processed_images": self.processed_images,
            "naming_regions": self.naming_regions,
            "localization_regions": self.localization_regions,
            "proposal_regions": self.proposal_regions,
            "sam_proposals_total": self.sam_proposals_total,
            "sam_empty_images": self.sam_empty_images,
            "corrclip_missing_mask_fallbacks": self.corrclip_missing_mask_fallbacks,
            "backbone": backbone,
            "prompt_ensemble": prompt_method,
            "localization_binarization": localization_threshold,
            "sam_checkpoint": str(SAM_CHECKPOINT),
            "sam_parameters": "points_per_side=32;pred_iou_thresh=0.86;stability_score_thresh=0.92;min_mask_region_area=400",
            "error_counts": json.dumps(dict(self.errors), sort_keys=True),
            "error_examples": " | ".join(self.error_examples),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=["context459", "ade847"])
    parser.add_argument("--num-images", type=int, default=250, help="First N manifest images per dataset (default: 250).")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def _import_corrclip():
    """Import CorrCLIP's vendored fork, ahead of pip's unmodified open_clip."""
    # ``_load_crop_clip`` intentionally imports pip open_clip first for its
    # normal global-image interface.  CorrCLIP has a same-named, incompatible
    # local fork whose VisionTransformer accepts DINO/mask inputs.  Evict only
    # those already-imported module objects before importing CorrCLIP; the crop
    # model itself remains valid because it holds references to its classes.
    for module_name in list(sys.modules):
        if module_name == "open_clip" or module_name.startswith("open_clip."):
            del sys.modules[module_name]
    sys.path.insert(0, str(CORRCLIP_ROOT))
    # The import is intentionally delayed until after the standard crop model
    # has been constructed.  CorrCLIP imports its modified local ``open_clip``.
    module = importlib.import_module("corrclip_segmentor")
    return module.CorrCLIPSegmentation


def _load_crop_clip(device: str):
    """Load the exact MetaCLIP FullCC weights into upstream open_clip.

    CorrCLIP modifies its vision tower to emit dense features requiring DINO
    inputs, so it cannot be called as a conventional global crop encoder.
    This standard global encoder has the same ViT-B/16-quickgelu architecture
    and loads the very same cached b16_fullcc2.5b checkpoint strictly.
    """
    import torch
    import open_clip

    if not METACLIP_CHECKPOINT.exists():
        raise FileNotFoundError(f"MetaCLIP checkpoint required for real crop naming is absent: {METACLIP_CHECKPOINT}")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained=None)
    checkpoint = torch.load(METACLIP_CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "MetaCLIP checkpoint did not strictly match open_clip ViT-B-16-quickgelu: "
            f"missing={incompatible.missing_keys[:5]}, unexpected={incompatible.unexpected_keys[:5]}"
        )
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-16-quickgelu")
    return model, preprocess, tokenizer


def _prompt_features(model, tokenizer, vocabulary: list[str], device: str):
    """CorrCLIP's own ImageNet prompt ensemble, averaged per class."""
    import torch

    # This import is from CorrCLIP's vendored repository and matches
    # CorrCLIPSegmentation.generate_category_embeddings exactly.
    from prompts.imagenet_template import openai_imagenet_template

    features = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
        for label in vocabulary:
            prompts = [template(label) for template in openai_imagenet_template]
            tokens = tokenizer(prompts).to(device)
            text = model.encode_text(tokens)
            text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            text = text.mean(dim=0)
            text = text / text.norm().clamp_min(1e-6)
            features.append(text)
    return torch.stack(features)


def _mask_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Return a bbox crop with all non-GT pixels blacked out."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty GT region")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    array = np.asarray(image).copy()
    array[~mask] = 0
    return Image.fromarray(array).crop((x1, y1, x2, y2))


def _naming_predictions(model, preprocess, text_features, image: Image.Image, regions: list[tuple[int, np.ndarray]], device: str):
    import torch

    crops = [_mask_crop(image, mask) for _, mask in regions]
    predictions: list[int] = []
    for start in range(0, len(crops), 32):
        tensor = torch.stack([preprocess(crop) for crop in crops[start : start + 32]]).to(device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            image_features = model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            predictions.extend((image_features @ text_features.T).argmax(dim=1).cpu().tolist())
    return predictions


def _resize_for_corrclip(image: Image.Image) -> Any:
    """Mirror CorrCLIP's test Resize(scale=(2048, 448), keep_ratio=True)."""
    import torch
    from torchvision.transforms import functional as TVF

    width, height = image.size
    scale = min(2048.0 / width, 448.0 / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    tensor = TVF.to_tensor(resized)
    # After CorrCLIP's MMseg data preprocessor (BGR->RGB then the following
    # normalization), its forward_feature's UnNormalize restores this tensor.
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _corrclip_crop_features(segmentor, crop, masks):
    """Exact CorrCLIP forward_feature up to (but excluding) text scoring."""
    import torch
    import torch.nn.functional as F

    imgs_norm = [segmentor.norm(segmentor.unnorm(crop[i])) for i in range(len(crop))]
    imgs_norm = torch.stack(imgs_norm, dim=0).half()
    segmentor.dino_qkv_output = None
    feat = segmentor.dino.get_intermediate_layers(imgs_norm, n=1)[-1]
    patch_size = segmentor.dino.patch_embed.patch_size
    feat_shape = (imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-1] // patch_size)
    batch, tokens = feat.shape[:2]
    qkv = segmentor.dino_qkv_output.reshape(batch, tokens, 3, -1).permute(2, 0, 1, 3)
    dino_feats = F.normalize(qkv[0] + qkv[1], dim=-1)[:, 1:]
    image_features = segmentor.clip.encode_image(crop.half(), dino_feats=dino_feats, feat_shape=feat_shape, instance_masks=masks)
    image_features = F.normalize(image_features, dim=-1)
    # B, L, D -> B, D, h, w.  This is where CorrCLIP's own forward_feature
    # would next take a dot product with query_features.
    return image_features.permute(0, 2, 1).reshape(batch, image_features.shape[-1], *feat_shape)


def _corrclip_feature_map(segmentor, normalized_image, instance_mask, original_shape: tuple[int, int]):
    """CorrCLIP forward_slide, but preserve text-independent dense features.

    Bilinear interpolation and the final text dot product are linear, so this
    is equivalent to running forward_slide separately with each single prompt,
    while avoiding re-running DINO/CLIP for every GT class.
    """
    import torch
    import torch.nn.functional as F

    image = normalized_image.unsqueeze(0).to(segmentor.device)
    masks = F.interpolate(instance_mask.unsqueeze(0).unsqueeze(0).float(), size=image.shape[-2:], mode="nearest").int()
    h_stride = w_stride = segmentor.slide_stride
    h_crop = w_crop = segmentor.slide_crop
    _, _, h_img, w_img = image.shape
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
    feats = None
    count = image.new_zeros((1, 1, h_img, w_img))
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1, x1 = h_idx * h_stride, w_idx * w_stride
            y2, x2 = min(y1 + h_crop, h_img), min(x1 + w_crop, w_img)
            y1, x1 = max(y2 - h_crop, 0), max(x2 - w_crop, 0)
            crop = image[:, :, y1:y2, x1:x2]
            crop_masks = masks[:, :, y1:y2, x1:x2]
            original_h, original_w = crop.shape[-2:]
            pad = segmentor.compute_padsize(original_h, original_w, 56)
            if any(pad):
                crop = F.pad(crop, pad)
                crop_masks = F.pad(crop_masks, pad, value=10000)
            crop_features = _corrclip_crop_features(segmentor, crop, crop_masks)
            crop_features = F.interpolate(crop_features, size=crop.shape[-2:], mode="bilinear")
            if any(pad):
                left, _, top, _ = pad
                crop_features = crop_features[:, :, top : top + original_h, left : left + original_w]
            if feats is None:
                feats = image.new_zeros((1, crop_features.shape[1], h_img, w_img))
            feats[:, :, y1:y2, x1:x2] += crop_features
            count[:, :, y1:y2, x1:x2] += 1
    if feats is None or bool((count == 0).any()):
        raise RuntimeError("CorrCLIP sliding-window feature coverage failed")
    feats = feats / count
    return F.interpolate(feats, size=original_shape, mode="bilinear")[0]


def _otsu_binary(score_map: np.ndarray) -> np.ndarray:
    """Image-only Otsu threshold for a one-prompt activation map."""
    finite = np.isfinite(score_map)
    if not finite.any():
        return np.zeros(score_map.shape, dtype=bool)
    values = score_map[finite]
    lo, hi = float(values.min()), float(values.max())
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return np.zeros(score_map.shape, dtype=bool)
    scaled = np.clip((score_map - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    hist = np.bincount(scaled[finite].ravel(), minlength=256).astype(np.float64)
    weight = hist.cumsum()
    mean = (hist * np.arange(256)).cumsum()
    total, total_mean = weight[-1], mean[-1]
    denominator = weight * (total - weight)
    between = np.divide((total_mean * weight - mean) ** 2, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    threshold = int(np.argmax(between))
    return scaled >= threshold


def _sam_best_ious(masks: list[dict[str, Any]], regions: list[tuple[int, np.ndarray]]) -> list[float]:
    """Best single real SAM mask IoU per semantic GT region."""
    if not masks:
        return [0.0] * len(regions)
    proposals = np.stack([np.asarray(item["segmentation"], dtype=bool) for item in masks], axis=0)
    proposal_flat = proposals.reshape(proposals.shape[0], -1)
    proposal_area = proposal_flat.sum(axis=1, dtype=np.int64)
    values: list[float] = []
    for _, gt_mask in regions:
        gt_flat = gt_mask.reshape(-1)
        intersection = proposal_flat[:, gt_flat].sum(axis=1, dtype=np.int64)
        union = proposal_area + int(gt_flat.sum()) - intersection
        values.append(float(np.max(np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0))))
    return values


def _build_sam(device: str):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM ViT-H checkpoint not found: {SAM_CHECKPOINT}")
    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device).eval()
    # Exactly the project's SamAmgClipAdapter parameters.
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=400,
    )


def run_dataset(spec: DatasetSpec, args: argparse.Namespace, crop_model, crop_preprocess, crop_tokenizer, segmentor, sam_generator) -> DiagnosticAccumulator:
    import torch
    from tf_ovos.data import load_manifest, read_vocab
    from tf_ovos.metrics import iou, load_label_map

    vocabulary = read_vocab(spec.vocab)
    if len(vocabulary) != spec.num_classes:
        raise ValueError(f"{spec.key}: expected {spec.num_classes} names, got {len(vocabulary)}")
    samples = load_manifest(spec.manifest)[: args.num_images]
    acc = DiagnosticAccumulator(spec=spec, requested_images=len(samples))
    text_features = _prompt_features(crop_model, crop_tokenizer, vocabulary, args.device)
    # Both models use the exact same checkpoint and ensemble.  The CorrCLIP
    # tensor is copied from the model itself to retain its dtype/device.
    corrclip_text = segmentor.query_features.detach()
    if corrclip_text.shape[0] != len(vocabulary):
        raise ValueError(f"{spec.key}: CorrCLIP query count {corrclip_text.shape[0]} != vocab {len(vocabulary)}")

    for index, sample in enumerate(samples, start=1):
        acc.attempted_images += 1
        try:
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            gt = load_label_map(str(sample.mask_path))
            if gt.shape != (image.height, image.width):
                raise ValueError(f"GT shape {gt.shape} differs from RGB image {(image.height, image.width)}")
            class_ids = np.unique(gt[(gt >= 0) & (gt < spec.num_classes)]).astype(int)
            regions = [(class_id, gt == class_id) for class_id in class_ids]
            if not regions:
                raise ValueError("no valid GT semantic classes")

            # 1. Real isolated-region naming.
            predicted_ids = _naming_predictions(crop_model, crop_preprocess, text_features, image, regions, args.device)
            acc.naming_regions += len(regions)
            acc.naming_hits += sum(int(pred == class_id) for pred, (class_id, _) in zip(predicted_ids, regions))

            # 2. Real CorrCLIP patch-correlation-refined one-prompt maps.
            expected_mask_file = spec.corrclip_masks / f"{Path(sample.image_path).stem}.npz"
            if not expected_mask_file.exists():
                acc.corrclip_missing_mask_fallbacks += 1
            instance_mask = segmentor.generate_mask(str(sample.image_path))
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                refined_features = _corrclip_feature_map(
                    segmentor,
                    _resize_for_corrclip(image),
                    instance_mask,
                    (image.height, image.width),
                )
                # NOTE: a single-prompt score map (this class's text vector
                # dotted with the dense features, Otsu-thresholded with no
                # other class competing) was tried first and rejected: its
                # raw similarity values span a narrow, low-contrast range
                # (~0.25-0.43) with no bimodal structure, so Otsu collapses
                # to a near-empty mask (<0.3% of the image) regardless of
                # image content -- verified on 8 samples, predicted area
                # 0.00-0.31% against GT areas of 3.5-22%. See plan.md
                # section 3.1 for the full writeup of that failure and this
                # fix. Scoring the target class against the full candidate
                # vocabulary and taking argmax==target as the predicted
                # region uses CorrCLIP's own real competitive normalization
                # instead of an ad-hoc single-prompt threshold, so it stays
                # well-calibrated; the tradeoff is that it no longer fully
                # isolates localization from naming competition (a class
                # that never wins the full argmax anywhere gets IoU=0 here
                # even if its patch-level correlation is locally reasonable).
                D, H, W = refined_features.shape
                flat = refined_features.reshape(D, -1).float()
                all_scores = corrclip_text.float() @ flat  # (num_classes, H*W)
                argmax_map = all_scores.argmax(dim=0).reshape(H, W).cpu().numpy()
                for class_id, gt_mask in regions:
                    predicted_mask = argmax_map == class_id
                    acc.localization_iou_sum += iou(predicted_mask, gt_mask)
                    acc.localization_regions += 1
                del all_scores, flat
            del refined_features
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            # 3+4. Independent, real SAM AMG proposals.
            sam_masks = sam_generator.generate(np.asarray(image))
            acc.sam_proposals_total += len(sam_masks)
            if not sam_masks:
                acc.sam_empty_images += 1
            best_ious = _sam_best_ious(sam_masks, regions)
            acc.proposal_regions += len(best_ious)
            acc.proposal_iou_sum += float(sum(best_ious))
            acc.proposal_recall_hits += sum(value >= 0.5 for value in best_ious)
            acc.processed_images += 1
        except Exception as exc:  # keep the pilot auditable instead of fabricating a result
            acc.note_error("image", sample.image_id, exc)
            print(f"[{spec.key} {index}/{len(samples)}] skipped {sample.image_id}: {type(exc).__name__}: {exc}", flush=True)
            if "CUDA out of memory" in str(exc) and args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        if index % args.progress_every == 0 or index == len(samples):
            print(
                f"[{spec.key} {index}/{len(samples)}] processed={acc.processed_images} "
                f"regions(n/l/p)={acc.naming_regions}/{acc.localization_regions}/{acc.proposal_regions} "
                f"sam_masks={acc.sam_proposals_total} errors={sum(acc.errors.values())}",
                flush=True,
            )
    return acc


def _average_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Observation 5's reported values are an unweighted average over datasets.
    keys = ["gt_region_naming_top1", "gt_text_localization_iou", "proposal_oracle_iou", "proposal_recall_at_05"]
    row: dict[str, Any] = {
        "row_type": "unweighted_dataset_mean",
        "dataset": "Context-459 + ADE-847 mean",
        "dataset_key": "mean",
        "method": "CorrCLIP",
        "diagnostic_source": "real_model_calls_pilot",
    }
    for key in keys:
        row[key] = float(np.mean([float(item[key]) for item in rows]))
    for key in rows[0]:
        row.setdefault(key, "")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]], accs: list[DiagnosticAccumulator], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = [
        ("GT-region naming Top-1", "gt_region_naming_top1", 0.487),
        ("GT-text localization IoU", "gt_text_localization_iou", 0.403),
        ("Proposal oracle IoU", "proposal_oracle_iou", 0.605),
        ("Proposal Recall@0.5", "proposal_recall_at_05", 0.635),
    ]
    dataset_rows, mean_row = rows[:-1], rows[-1]
    lines = [
        "# CorrCLIP real Observation-5 diagnostic pilot",
        "",
        "This is a real-call pilot, not a replacement for the existing dense-label-map proxy table. It evaluates CorrCLIP only on the first "
        f"{args.num_images} manifest records per requested dataset.",
        "",
        "## Results",
        "",
        "| Metric | Context-459 | ADE-847 | Unweighted mean | Old six-dataset proxy | Mean minus proxy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, key, proxy in metrics:
        values = [float(row[key]) for row in dataset_rows]
        context = values[[row["dataset_key"] for row in dataset_rows].index("context459")] if any(row["dataset_key"] == "context459" for row in dataset_rows) else float("nan")
        ade = values[[row["dataset_key"] for row in dataset_rows].index("ade847")] if any(row["dataset_key"] == "ade847" for row in dataset_rows) else float("nan")
        mean = float(mean_row[key])
        lines.append(f"| {label} | {context:.3f} | {ade:.3f} | {mean:.3f} | {proxy:.3f} | {mean - proxy:+.3f} |")
    lines.extend([
        "",
        "The old values (0.487 / 0.403 / 0.605 / 0.635) are the disclosed proxy values: an unweighted six-dataset average reconstructed from CorrCLIP dense argmax maps. They are not directly like-for-like with this two-dataset, real-call pilot.",
        "",
        "## What was actually run",
        "",
        "- **Backbone:** MetaCLIP FullCC `ViT-B-16-quickgelu`, using CorrCLIP's cached `b16_fullcc2.5b.pt` weights. Region naming uses a conventional global open_clip ViT-B/16-quickgelu encoder loaded strictly from that same checkpoint; CorrCLIP's vendored vision tower cannot serve a global crop embedding because it is modified to require DINO correlation inputs. Localization uses CorrCLIP's vendored DINO ViT-B/8 + correlation-refined dense encoder directly.",
        "- **Prompts:** CorrCLIP's own `openai_imagenet_template` ensemble, mean-normalized per dataset vocabulary class. The repository's CorrCLIP vocabulary files match the benchmark vocabularies byte-for-byte apart from their final newline.",
        "- **Naming:** for each present semantic class, GT pixels outside that class are blacked out, the tight bounding-box crop is CLIP-encoded, and its cosine top-1 is evaluated over the full fixed dataset vocabulary.",
        "- **Localization:** the correlation-refined dense image features are computed once per image, then scored against the *full* candidate vocabulary (not just the one correct class). The predicted region for a given GT class is the set of pixels where that class wins the argmax against every other candidate class, using CorrCLIP's own real dense scores (freshly computed here, not read from a saved prediction map). An earlier version of this pilot scored only the single correct class's prompt and Otsu-thresholded that one map; it was rejected after a manual check showed predicted regions collapsing to under 0.3% of the image regardless of GT size (3.5-22%), because a single-prompt similarity map has too narrow/unimodal a value range for Otsu to split meaningfully. The argmax version does not fully isolate localization from naming competition -- a class that never wins the full argmax anywhere gets IoU=0 here even if CorrCLIP's patch-level correlation for it is locally reasonable -- but it is a real, non-degenerate measurement rather than a broken one.",
        "- **Proposals:** SAM ViT-H automatic-mask generation from `weights/sam_vit_h_4b8939.pth`, with the exact parameters used by `SamAmgClipAdapter`: points-per-side 32, predicted-IoU threshold 0.86, stability threshold 0.92, and min region area 400. For every semantic GT region, oracle IoU is the largest IoU of one generated SAM mask; Recall@0.5 is the fraction with best IoU >= 0.5.",
        "",
        "## Coverage and failures",
        "",
        "| Dataset | Requested | Attempted | Fully processed | Naming regions | Localization regions | Proposal regions | SAM masks | Empty-SAM images | CorrCLIP saved-mask fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for acc in accs:
        lines.append(
            f"| {acc.spec.display_name} | {acc.requested_images} | {acc.attempted_images} | {acc.processed_images} | "
            f"{acc.naming_regions} | {acc.localization_regions} | {acc.proposal_regions} | {acc.sam_proposals_total} | "
            f"{acc.sam_empty_images} | {acc.corrclip_missing_mask_fallbacks} |"
        )
    for acc in accs:
        if acc.errors:
            lines.extend(["", f"{acc.spec.display_name} skipped {sum(acc.errors.values())} image(s): `{dict(acc.errors)}`."])
            lines.extend([f"- `{example}`" for example in acc.error_examples])
    lines.extend([
        "",
        "## Caveats",
        "",
        f"This pilot is limited to up to {args.num_images} images per dataset (first manifest order, capped at each dataset's actual validation-set size), two large-vocabulary datasets, and CorrCLIP alone. Semantic regions are class unions, not separately annotated object instances. The proposal results measure generic SAM ViT-H coverage, not CorrCLIP's saved SAM2 masks or a learned proposal method. The localization result uses full-vocabulary argmax (see above) rather than a pure single-class score, so it is not a fully naming-independent localization measurement -- classes that never win the argmax anywhere get IoU=0 by construction. These measurements establish that the four quantities were obtained with actual CLIP/CorrCLIP/SAM computations on this subset; they do not establish method-general or paper-final claims beyond the two datasets evaluated here.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    # Import project modules only after startup checks, so a hard dependency
    # failure is explicit rather than accidentally yielding proxy numbers.
    sys.path.insert(0, str(ROOT / "src"))
    crop_model, crop_preprocess, crop_tokenizer = _load_crop_clip(args.device)
    CorrCLIPSegmentation = _import_corrclip()
    segmentor = CorrCLIPSegmentation(
        clip_type="metaclip_fullcc",
        model_type="ViT-B-16-quickgelu",
        dino_type="dino_vitb8",
        name_path=str(SPECS[args.datasets[0]].corrclip_names),
        device=__import__("torch").device(args.device),
        instance_mask_path=str(SPECS[args.datasets[0]].corrclip_masks),
        mask_generator=None,
    )
    sam_generator = _build_sam(args.device)
    accs = []
    for key in args.datasets:
        spec = SPECS[key]
        # The model's category embeddings and pre-generated mask directory are
        # dataset-specific, just as they are in CorrCLIP's official configs.
        if key != args.datasets[0]:
            segmentor.generate_category_embeddings(str(spec.corrclip_names), segmentor.device)
            segmentor.instance_mask_path = str(spec.corrclip_masks)
        accs.append(run_dataset(spec, args, crop_model, crop_preprocess, crop_tokenizer, segmentor, sam_generator))
    backbone = "metaclip_fullcc; ViT-B-16-quickgelu; CorrCLIP DINO ViT-B/8 dense localization"
    prompt_method = "CorrCLIP openai_imagenet_template ensemble (mean-normalized)"
    threshold = (
        "full-vocabulary argmax over CorrCLIP's own dense scores (target-class "
        "region = pixels where target class wins argmax against all other "
        "candidate classes); single-prompt Otsu binarization was tried first "
        "and rejected -- collapses to near-empty masks (<0.3% area vs 3.5-22% "
        "GT area) on a low-contrast score range, see plan.md 3.1"
    )
    rows = [acc.row(backbone, prompt_method, threshold) for acc in accs]
    rows.append(_average_row(rows))
    _write_csv(args.output_csv, rows)
    _write_markdown(args.output_md, rows, accs, args)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
