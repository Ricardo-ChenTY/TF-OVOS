#!/usr/bin/env python
"""Real, bounded Observation-5 diagnostic pilot for FreeDA.

Separate from ``real_obs5_diagnostic_multi.py`` because FreeDA requires its
own Python 3.8 environment (``freeda-official-py38-t112b``, pinned
mmcv/mmseg 0.x) and a fundamentally different input tensor convention (raw
BGR pixel values in 0-255 range, not CLIP-normalized RGB) -- trying to share
one script/process across both would fight the two codebases' assumptions
rather than reduce risk.

This reuses FreeDA's own real config-loading, model-building, and mmseg
EncoderDecoder inference code path (``FreeDASegInference``, inherited
unmodified ``slide_inference``) exactly as the official
``run_freeda_cliptrase_repair_queue.sh`` evaluation does, rather than
reimplementing the sliding-window/background-channel logic by hand. Only the
GT-region naming crop loop and SAM proposal loop are added on top, mirroring
the same three probes as ``real_obs5_diagnostic_multi.py``:

1. GT-region naming: blacked-out bbox crop of each GT class region, encoded
   by FreeDA's own ``clip_model`` (ViT-L-14, openai), cosine top-1 against
   the full vocabulary's mean-prompt-ensemble text embedding (the same
   ``text_embedding`` FreeDA's own ``build_proto_embedding`` already
   produces for the real leaderboard run).
2. GT-text localization: full-vocabulary argmax over FreeDA's own real
   ``FreeDASegInference.inference()`` output (real DINOv2 backbone +
   superpixel proposals + diffusion-collection prototype retrieval +
   CLIP-similarity ensemble, genuinely computed per image, not read from a
   saved prediction).
3. Proposal oracle/recall: SAM ViT-H automatic mask generation, identical
   parameters to every other method's probe in this project
   (points_per_side=32, pred_iou_thresh=0.86, stability_score_thresh=0.92,
   min_mask_region_area=400).

Must be run with the FreeDA conda environment, from any cwd (the script
itself chdirs into FreeDA's ``src`` directory, which its own relative config
paths and imports require):

  /data/tianyi/conda_envs/freeda-official-py38-t112b/bin/python \\
      scripts/real_obs5_diagnostic_freeda.py --datasets context459 ade847 --num-images 5105
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path("/data/tianyi/TF-OVCOS")
FREEDA_SRC = ROOT / "third_party" / "official_methods" / "freeda" / "src"
SAM_CHECKPOINT = ROOT / "weights" / "sam_vit_h_4b8939.pth"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    manifest: Path
    vocab: Path
    num_classes: int
    eval_cfg: str  # relative to FREEDA_SRC
    seg_cfg_module: str  # dotted, e.g. "pascal_context459"
    template: str


SPECS = {
    "context459": DatasetSpec(
        key="context459",
        display_name="Context-459",
        manifest=ROOT / "data" / "manifests" / "context459_val.jsonl",
        vocab=ROOT / "configs" / "vocab" / "context_459.txt",
        num_classes=459,
        eval_cfg="configs/pascal59/freeda_pascal59.yml",
        seg_cfg_module="segmentation.configs._base_.datasets.pascal_context459",
        template="sub_imagenet_template",
    ),
    "ade847": DatasetSpec(
        key="ade847",
        display_name="ADE-847",
        manifest=ROOT / "data" / "manifests" / "ade20k847_val.jsonl",
        vocab=ROOT / "configs" / "vocab" / "ade20k_847.txt",
        num_classes=847,
        eval_cfg="configs/ade/freeda_ade.yml",
        seg_cfg_module="segmentation.configs._base_.datasets.ade20k847",
        template="sub_imagenet_template",
    ),
}


def _read_vocab(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _build_model_and_seg(spec: DatasetSpec, device: str):
    """Mirror main.py's --eval config merge and model build exactly, then
    wrap in FreeDASegInference the same way build_freeda_seg_inference does
    -- but with classnames taken from this project's own canonical vocab
    file instead of building a full mmseg dataset object (the benchmark's
    fixed protocol guarantees this project's vocab.txt already matches the
    class list/order FreeDA's own dataset registration would give)."""
    from omegaconf import OmegaConf
    from utils import load_config
    from models import build_model
    from segmentation.evaluation.freeda_seg import FreeDASegInference

    default_cfg = load_config(spec.eval_cfg)
    org_cfg = OmegaConf.create()
    eval_cfg = OmegaConf.create(
        {
            "evaluate": {
                "pamr": False,
                "bg_thresh": 0.4,
                "kp_w": 0.3,
                "pred_qual_path": None,
                "gt_qual_path": None,
                "eval_only": True,
                "template": spec.template,
            }
        }
    )
    cfg = OmegaConf.merge(default_cfg, org_cfg, eval_cfg)
    cfg.model.into_the_wild = False

    model = build_model(cfg.model)
    model.eval().to(device)

    classnames = _read_vocab(spec.vocab)
    if len(classnames) != spec.num_classes:
        raise ValueError(f"{spec.key}: expected {spec.num_classes} vocab names, got {len(classnames)}")

    text_tokens = model.build_dataset_class_tokens(cfg.evaluate.template, classnames)
    text_embedding, proto_embedding = model.build_proto_embedding(text_tokens)

    seg_model = FreeDASegInference(
        model,
        text_embedding,
        proto_embedding,
        classnames,
        with_bg=False,  # this benchmark's Context-459/ADE-847 vocabs have no leading "background" class
        test_cfg=dict(mode="slide", stride=(224, 224), crop_size=(448, 448)),
        **cfg.evaluate,
    )
    seg_model.eval().to(device)
    return model, seg_model, classnames, text_embedding


def _resize_keep_ratio_bgr255(image: Image.Image, img_scale=(2048, 448)):
    """Mirror this project's FreeDA test_pipeline: LoadImageFromFile (BGR,
    0-255) -> Resize(keep_ratio=True) -> FloatImage -> ImageToTensor. No
    Normalize step exists in FreeDA's own pipeline -- generate_masks does its
    own BGR->RGB swap and CLIP/DINOv2-specific normalization internally."""
    import torch

    width, height = image.size
    long_edge, short_edge = img_scale
    ratio = min(long_edge / max(width, height), short_edge / min(width, height))
    new_w, new_h = max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    rgb = np.asarray(resized, dtype=np.float32)  # H, W, 3 (RGB, 0-255)
    bgr = rgb[:, :, ::-1].copy()  # LoadImageFromFile loads BGR
    tensor = torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    return tensor, (new_h, new_w)


def _clip_preprocess_transform():
    from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

    return Compose([Resize(224, interpolation=InterpolationMode.BICUBIC), CenterCrop(224)])


def _mask_crop(image: Image.Image, mask: np.ndarray) -> Image.Image:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("empty GT region")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    array = np.asarray(image).copy()
    array[~mask] = 0
    return Image.fromarray(array).crop((x1, y1, x2, y2))


def _sam_best_ious(masks: list[dict[str, Any]], regions: list[tuple[int, np.ndarray]]) -> list[float]:
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
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=400,
    )


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
    errors: Counter = field(default_factory=Counter)
    error_examples: list = field(default_factory=list)
    sam_proposals_total: int = 0
    sam_empty_images: int = 0

    def note_error(self, stage, sample_id, exc):
        key = f"{stage}:{type(exc).__name__}"
        self.errors[key] += 1
        if len(self.error_examples) < 8:
            self.error_examples.append(f"{sample_id} [{stage}]: {type(exc).__name__}: {exc}")

    def row(self, method: str):
        denom_naming = max(self.naming_regions, 1)
        denom_loc = max(self.localization_regions, 1)
        denom_prop = max(self.proposal_regions, 1)
        return {
            "row_type": "dataset",
            "dataset": self.spec.display_name,
            "dataset_key": self.spec.key,
            "method": method,
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
            "backbone": "FreeDA's own real pipeline: DINOv2-L/14 + superpixel proposals + diffusion-collection prototype retrieval + CLIP ViT-L-14 (openai) ensemble (FreeDASegInference.inference, real slide-window)",
            "prompt_ensemble": f"FreeDA's own {self.spec.template} template, mean-normalized" if hasattr(self.spec, "template") else "sub_imagenet_template",
            "localization_binarization": "full-vocabulary argmax over FreeDA's own real FreeDASegInference.inference() softmax output (real slide-window, pamr off, bg_thresh 0.4 unused since with_bg=False)",
            "sam_checkpoint": str(SAM_CHECKPOINT),
            "sam_parameters": "points_per_side=32;pred_iou_thresh=0.86;stability_score_thresh=0.92;min_mask_region_area=400",
            "error_counts": json.dumps(dict(self.errors), sort_keys=True),
            "error_examples": " | ".join(self.error_examples),
        }


def run_dataset(spec: DatasetSpec, device: str, num_images: int, progress_every: int) -> DiagnosticAccumulator:
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from tf_ovos.data import load_manifest
    from tf_ovos.metrics import iou, load_label_map

    model, seg_model, classnames, text_embedding = _build_model_and_seg(spec, device)
    clip_preprocess = _clip_preprocess_transform()
    sam_generator = _build_sam(device)

    samples = load_manifest(spec.manifest)[:num_images]
    acc = DiagnosticAccumulator(spec=spec, requested_images=len(samples))

    for index, sample in enumerate(samples, start=1):
        acc.attempted_images += 1
        try:
            with Image.open(sample.image_path) as source:
                image = source.convert("RGB")
            gt = load_label_map(str(sample.mask_path))
            if gt.shape != (image.height, image.width):
                raise ValueError(f"GT shape {gt.shape} differs from RGB image {(image.height, image.width)}")
            class_ids = np.unique(gt[(gt >= 0) & (gt < spec.num_classes)]).astype(int)
            regions = [(c, gt == c) for c in class_ids]
            if not regions:
                raise ValueError("no valid GT semantic classes")

            # 1. Real GT-region naming via FreeDA's own clip_model.
            crops = [_mask_crop(image, mask) for _, mask in regions]
            predicted_ids = []
            with torch.no_grad():
                for start in range(0, len(crops), 16):
                    batch_crops = crops[start : start + 16]
                    tensors = torch.stack(
                        [torch.from_numpy(np.asarray(clip_preprocess(c), dtype=np.float32)).permute(2, 0, 1) for c in batch_crops]
                    ).to(device)
                    feats = model.clip_model.encode_image(model.clip_image_preprocess(tensors))
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    sims = (feats.float() @ text_embedding.float().T).cpu().numpy()
                    predicted_ids.extend(np.argmax(sims, axis=1).tolist())
            acc.naming_regions += len(regions)
            acc.naming_hits += sum(int(p == c) for p, (c, _) in zip(predicted_ids, regions))

            # 2. Real dense localization via FreeDA's own FreeDASegInference.
            tensor, (h, w) = _resize_keep_ratio_bgr255(image)
            tensor = tensor.to(device)
            img_meta = [
                {
                    "ori_shape": (image.height, image.width, 3),
                    "img_shape": (h, w, 3),
                    "pad_shape": (h, w, 3),
                    "flip": False,
                    "flip_direction": "horizontal",
                    "scale_factor": 1.0,
                    "ori_filename": Path(sample.image_path).name,
                }
            ]
            with torch.no_grad():
                probs = seg_model.inference(tensor, img_meta, rescale=True)
            argmax_map = probs[0].argmax(dim=0).cpu().numpy()
            for class_id, gt_mask in regions:
                predicted_mask = argmax_map == class_id
                acc.localization_iou_sum += iou(predicted_mask, gt_mask)
                acc.localization_regions += 1

            # 3. Real, independent SAM AMG proposals (identical params project-wide).
            sam_masks = sam_generator.generate(np.asarray(image))
            acc.sam_proposals_total += len(sam_masks)
            if not sam_masks:
                acc.sam_empty_images += 1
            best_ious = _sam_best_ious(sam_masks, regions)
            acc.proposal_regions += len(best_ious)
            acc.proposal_iou_sum += float(sum(best_ious))
            acc.proposal_recall_hits += sum(v >= 0.5 for v in best_ious)
            acc.processed_images += 1
        except Exception as exc:
            acc.note_error("image", sample.image_id, exc)
            print(f"[{spec.key} {index}/{len(samples)}] skipped {sample.image_id}: {type(exc).__name__}: {exc}", flush=True)
            if "CUDA out of memory" in str(exc):
                torch.cuda.empty_cache()
        if index % progress_every == 0 or index == len(samples):
            print(
                f"[{spec.key} {index}/{len(samples)}] processed={acc.processed_images} "
                f"regions(n/l/p)={acc.naming_regions}/{acc.localization_regions}/{acc.proposal_regions} "
                f"sam_masks={acc.sam_proposals_total} errors={sum(acc.errors.values())}",
                flush=True,
            )
    return acc


def _average_row(rows):
    keys = ["gt_region_naming_top1", "gt_text_localization_iou", "proposal_oracle_iou", "proposal_recall_at_05"]
    row = {
        "row_type": "unweighted_dataset_mean",
        "dataset": "Context-459 + ADE-847 mean",
        "dataset_key": "mean",
        "method": "freeda",
        "diagnostic_source": "real_model_calls_pilot",
    }
    for key in keys:
        row[key] = float(np.mean([float(item[key]) for item in rows]))
    for key in rows[0]:
        row.setdefault(key, "")
    return row


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=["context459", "ade847"])
    parser.add_argument("--num-images", type=int, default=250)
    parser.add_argument("--output-csv", type=Path, default=ROOT / "runs" / "analysis" / "obs5_real_diagnostic_freeda.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    os.chdir(FREEDA_SRC)
    sys.path.insert(0, str(FREEDA_SRC))

    rows = []
    for key in args.datasets:
        spec = SPECS[key]
        acc = run_dataset(spec, args.device, args.num_images, args.progress_every)
        rows.append(acc.row("freeda"))
        import torch

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    rows.append(_average_row(rows))
    _write_csv(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
