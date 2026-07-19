#!/usr/bin/env python
"""Real, bounded Observation-5 diagnostic pilot, extended to multiple methods.

``scripts/real_obs5_diagnostic_pilot.py`` implements this for CorrCLIP only.
This script generalizes the same three real-inference probes (GT-region
naming, GT-text localization via full-vocabulary argmax, SAM proposal
oracle/recall) to additional training-free methods, so Observation 5's
disclosed proxy table can be checked against more than one method.

Each method is a self-contained "backend" that knows how to (a) produce a
dense per-pixel class-score map for the full candidate vocabulary using that
method's own real forward pass, and (b) produce a plain global crop
embedding for the GT-region naming probe. The proposal probe (SAM ViT-H
automatic mask generation) is method-agnostic and identical for every
backend, exactly mirroring ``SamAmgClipAdapter``'s parameters.

Typical invocation (the project environment is required):

  /data/tianyi/conda_envs/tf-ovos/bin/python scripts/real_obs5_diagnostic_multi.py \\
      --method naclip --datasets context459 ade847 --num-images 5105

Add ``--num-images 250`` for a bounded pilot instead of the full validation
set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SAM_CHECKPOINT = ROOT / "weights" / "sam_vit_h_4b8939.pth"

# Captured at true module-import time, before any backend inserts a vendored
# method directory at the front of sys.path and evicts 'clip' from
# sys.modules. This is the plain pip-installed OpenAI CLIP package (already
# present in this environment), genuinely unmodified by any method's own
# vendored architecture surgery. Methods whose own vendored vision tower has
# no "vanilla" bypass (SC-CLIP's LOF/self-correlation pipeline runs
# unconditionally, unlike NACLIP's set_params escape hatch) must use this
# for the naming probe's global crop embedding instead of re-loading their
# own vendored ``clip`` package, which would just re-run the same surgery.
import clip as _stock_openai_clip  # noqa: E402
import open_clip as _stock_open_clip  # noqa: E402

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    manifest: Path
    vocab: Path
    num_classes: int


# The vendored methods' own `configs/cls_*.txt` files use this dataset key
# spelling (e.g. `cls_ade20k847.txt`), which differs from this project's own
# dataset key ("ade847").
CLS_FILE_KEY = {"context459": "context459", "ade847": "ade20k847"}


SPECS = {
    "context459": DatasetSpec(
        key="context459",
        display_name="Context-459",
        manifest=ROOT / "data" / "manifests" / "context459_val.jsonl",
        vocab=ROOT / "configs" / "vocab" / "context_459.txt",
        num_classes=459,
    ),
    "ade847": DatasetSpec(
        key="ade847",
        display_name="ADE-847",
        manifest=ROOT / "data" / "manifests" / "ade20k847_val.jsonl",
        vocab=ROOT / "configs" / "vocab" / "ade20k_847.txt",
        num_classes=847,
    ),
}


def _deregister_mmseg(*names: str) -> None:
    """Remove a stale mmengine MODELS registration before re-importing a
    vendored segmentor module for a second dataset in the same process.
    Each backend's __init__ re-executes its module (via ``_evict`` forcing a
    fresh import), which re-runs that module's ``@MODELS.register_module()``
    class decorator -- but mmengine's registry is a process-global singleton
    keyed by class name, so a second registration of the same name raises
    KeyError unless the stale entry is cleared first."""
    from mmseg.registry import MODELS

    for name in names:
        MODELS._module_dict.pop(name, None)


def _evict(prefixes: list[str]) -> None:
    """Evict already-imported modules whose name matches one of the given
    prefixes, so a different method's same-named vendored module (``clip``,
    ``prompts``, ``custom_datasets``, ...) can be imported cleanly instead."""
    for module_name in list(sys.modules):
        if any(module_name == p or module_name.startswith(p + ".") for p in prefixes):
            del sys.modules[module_name]


def _resize_keep_ratio(image: Image.Image, scale: tuple[int, int]) -> Any:
    """Mirror mmseg's ``Resize(scale=scale, keep_ratio=True)`` + CLIP-style
    normalization, exactly as ``real_obs5_diagnostic_pilot.py`` does for
    CorrCLIP. ``scale`` is ``(max_long_side, max_short_side)`` per mmseg
    convention: the resize factor is the minimum ratio that fits both."""
    import torch
    from torchvision.transforms import functional as TVF

    width, height = image.size
    long_edge, short_edge = scale
    ratio = min(long_edge / max(width, height), short_edge / min(width, height))
    new_size = (max(1, int(round(width * ratio))), max(1, int(round(height * ratio))))
    resized = image.resize(new_size, Image.Resampling.BILINEAR)
    tensor = TVF.to_tensor(resized)
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor(CLIP_STD, dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


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


class MethodBackend:
    """Interface every per-method backend must satisfy."""

    name: str
    backbone_desc: str
    prompt_desc: str
    localization_desc: str

    def dense_argmax_map(self, image: Image.Image, image_path: str) -> np.ndarray:
        """Return an (H, W) int array: the full-vocabulary argmax class id
        at every pixel, at the image's *original* resolution, using this
        method's own real dense forward pass (its actual leaderboard
        inference path, not a generic encoder). ``image_path`` is passed
        alongside the already-opened PIL image because some methods (e.g.
        Trident) re-read the raw file for a second, independently-processed
        input stream (SAM feature extraction)."""
        raise NotImplementedError

    def naming_predictions(self, image: Image.Image, regions: list[tuple[int, np.ndarray]]) -> list[int]:
        """Return the predicted class id for each GT region, using a plain
        global crop embedding against the full vocabulary."""
        raise NotImplementedError


class NaclipBackend(MethodBackend):
    name = "naclip"
    backbone_desc = "OpenAI CLIP ViT-B/16 (NACLIP's own vendored checkpoint), NACLIP dense inference with PAMR"
    prompt_desc = "NACLIP's own openai_imagenet_template ensemble (mean-normalized)"
    localization_desc = (
        "full-vocabulary argmax over NACLIP's own real dense scores (forward_slide + "
        "synonym-merge + PAMR, target-class region = pixels where target class wins "
        "argmax against all other candidate classes)"
    )
    resize_scale = (2048, 448)

    def __init__(self, spec: DatasetSpec, device: str):
        method_root = ROOT / "third_party" / "official_methods" / "NACLIP"
        _deregister_mmseg("NACLIP")
        _evict(["clip", "prompts", "naclip", "custom_datasets", "pamr"])
        sys.path.insert(0, str(method_root))
        import clip as naclip_clip
        from naclip import NACLIP, get_cls_idx

        self._clip_module = naclip_clip
        self.device = device
        name_path = method_root / "configs" / f"cls_{CLS_FILE_KEY[spec.key]}.txt"
        self.segmentor = NACLIP(clip_path="ViT-B/16", name_path=str(name_path), device=device)
        self.net = self.segmentor.net
        self._reduced_params = ("reduced", "naclip", 5.0)
        self._vanilla_params = ("vanilla", "vanilla", 5.0)
        self.net.visual.set_params(*self._reduced_params)

        # A dedicated, vanilla text-feature bank over the canonical vocab.txt
        # classes (459/847-way), independent of NACLIP's internal
        # synonym-expanded query list, so naming is scored on the same axis
        # as the localization probe's class ids.
        import torch
        from prompts.imagenet_template import openai_imagenet_template

        vocabulary = _read_vocab(spec.vocab)
        if len(vocabulary) != spec.num_classes:
            raise ValueError(f"{spec.key}: expected {spec.num_classes} vocab names, got {len(vocabulary)}")
        self.vocabulary = vocabulary
        features = []
        with torch.no_grad():
            for label in vocabulary:
                tokens = naclip_clip.tokenize([t(label) for t in openai_imagenet_template]).to(device)
                text = self.net.encode_text(tokens)
                text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                text = text.mean(dim=0)
                text = text / text.norm().clamp_min(1e-6)
                features.append(text)
        self.text_features = torch.stack(features)

        # get_cls_idx / query_idx let us collapse NACLIP's own internal
        # synonym-duplicated queries down to one score per canonical class,
        # exactly like NACLIP.postprocess_result does for the real leaderboard.
        self.query_idx = self.segmentor.query_idx
        self.num_cls = int(self.query_idx.max().item()) + 1
        if self.num_cls != spec.num_classes:
            raise ValueError(f"{spec.key}: NACLIP query_idx implies {self.num_cls} classes, vocab has {spec.num_classes}")

    def dense_argmax_map(self, image: Image.Image, image_path: str) -> np.ndarray:
        import torch
        from naclip import _merge_synonym_probs

        tensor = _resize_keep_ratio(image, self.resize_scale).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.segmentor.forward_slide(tensor, self.segmentor.slide_stride, self.segmentor.slide_crop)
            logits = torch.nn.functional.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            probs = logits[0]
            if self.num_cls != probs.shape[0]:
                probs = _merge_synonym_probs(probs, self.query_idx, self.num_cls, probs.shape[0])
            if self.segmentor.pamr is not None:
                img_resized = torch.nn.functional.interpolate(
                    tensor, size=(image.height, image.width), mode="bilinear", align_corners=False
                )
                try:
                    probs = self.segmentor.pamr(img_resized, probs.unsqueeze(0).to(img_resized.dtype))[0].to(probs.dtype)
                except RuntimeError:
                    pass  # mirror NACLIP's own PAMR-OOM fallback: skip refinement for this image
            argmax_map = probs.argmax(dim=0).cpu().numpy()
        return argmax_map

    def naming_predictions(self, image: Image.Image, regions: list[tuple[int, np.ndarray]]) -> list[int]:
        import torch

        crops = [_mask_crop(image, mask) for _, mask in regions]
        preprocess = _clip_preprocess_transform()
        predictions: list[int] = []
        self.net.visual.set_params(*self._vanilla_params)
        try:
            for start in range(0, len(crops), 32):
                batch = torch.stack([preprocess(c) for c in crops[start : start + 32]]).to(self.device)
                with torch.inference_mode():
                    feats = self.net.encode_image(batch)
                    feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                    predictions.extend((feats.float() @ self.text_features.float().T).argmax(dim=1).cpu().tolist())
        finally:
            self.net.visual.set_params(*self._reduced_params)
        return predictions


class ScclipBackend(MethodBackend):
    name = "scclip"
    backbone_desc = "OpenAI CLIP ViT-B/16 (SC-CLIP's own vendored checkpoint) + a second unmodified CLIP instance for naming"
    prompt_desc = "SC-CLIP's own openai_imagenet_template ensemble (mean-normalized)"
    localization_desc = (
        "full-vocabulary argmax over SC-CLIP's own real dense scores (forward_slide "
        "self-correlation pipeline + synonym-merge; PAMR left off, matching this "
        "project's actual SC-CLIP leaderboard config)"
    )
    resize_scale = (2048, 336)

    def __init__(self, spec: DatasetSpec, device: str):
        method_root = ROOT / "third_party" / "official_methods" / "SC-CLIP"
        _deregister_mmseg("SCCLIPForSegmentation")
        _evict(["clip", "prompts", "scclip_segmentor", "custom_datasets", "pamr"])
        sys.path.insert(0, str(method_root))
        import clip as scclip_clip
        from scclip_segmentor import SCCLIPForSegmentation

        self.device = device
        name_path = method_root / "configs" / f"cls_{CLS_FILE_KEY[spec.key]}.txt"
        self.segmentor = SCCLIPForSegmentation(clip_path="ViT-B/16", name_path=str(name_path), device=device)

        # SC-CLIP's own vendored VisionTransformer.forward runs its
        # LOF-outlier/self-correlation surgery unconditionally, with no
        # vanilla bypass (unlike NACLIP's set_params escape hatch) -- even a
        # second instance loaded via SC-CLIP's own vendored `clip.load` would
        # still crash the same way (confirmed empirically: `lof_pytorch`
        # raises a CUDA index-out-of-bounds assert on a real crop). Naming
        # therefore needs the genuinely unmodified pip `clip` package
        # captured at module-import time, not SC-CLIP's own vendored copy.
        self.naming_net, _ = _stock_openai_clip.load("ViT-B/16", device=device, jit=False)
        self.naming_net.eval()

        import torch
        from prompts.imagenet_template import openai_imagenet_template

        vocabulary = _read_vocab(spec.vocab)
        if len(vocabulary) != spec.num_classes:
            raise ValueError(f"{spec.key}: expected {spec.num_classes} vocab names, got {len(vocabulary)}")
        self.vocabulary = vocabulary
        features = []
        with torch.no_grad():
            for label in vocabulary:
                tokens = _stock_openai_clip.tokenize([t(label) for t in openai_imagenet_template]).to(device)
                text = self.naming_net.encode_text(tokens)
                text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                text = text.mean(dim=0)
                text = text / text.norm().clamp_min(1e-6)
                features.append(text)
        self.text_features = torch.stack(features)

        self.query_idx = self.segmentor.query_idx
        self.num_cls = self.segmentor.num_classes
        if self.num_cls != spec.num_classes:
            raise ValueError(f"{spec.key}: SC-CLIP query_idx implies {self.num_cls} classes, vocab has {spec.num_classes}")

    def dense_argmax_map(self, image: Image.Image, image_path: str) -> np.ndarray:
        import torch

        tensor = _resize_keep_ratio(image, self.resize_scale).unsqueeze(0).to(self.device)
        img_metas = [{"ori_shape": (image.height, image.width)}]
        with torch.inference_mode():
            logits = self.segmentor.forward_slide(tensor, img_metas, self.segmentor.slide_stride, self.segmentor.slide_crop)
            probs = (logits[0] * self.segmentor.logit_scale).softmax(0)
            if self.num_cls != probs.shape[0]:
                class_probs = probs.new_zeros((self.num_cls, *probs.shape[1:]))
                index = self.query_idx.view(-1, 1, 1).expand_as(probs)
                class_probs.scatter_reduce_(0, index, probs, reduce="amax", include_self=True)
                probs = class_probs
            argmax_map = probs.argmax(dim=0).cpu().numpy()
        return argmax_map

    def naming_predictions(self, image: Image.Image, regions: list[tuple[int, np.ndarray]]) -> list[int]:
        import torch

        crops = [_mask_crop(image, mask) for _, mask in regions]
        preprocess = _clip_preprocess_transform()
        predictions: list[int] = []
        for start in range(0, len(crops), 32):
            batch = torch.stack([preprocess(c) for c in crops[start : start + 32]]).to(self.device)
            with torch.inference_mode():
                feats = self.naming_net.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                predictions.extend((feats.float() @ self.text_features.float().T).argmax(dim=1).cpu().tolist())
        return predictions


class TridentBackend(MethodBackend):
    name = "trident"
    backbone_desc = (
        "Trident's own real pipeline: OpenAI CLIP ViT-B/16 + DINO ViT-B/16 + SAM ViT-B window-guidance "
        "(get_trident_seg, sam_refinement off); naming uses a second, separately-loaded stock CLIP ViT-B/16"
    )
    prompt_desc = "Trident's own openai_imagenet_template ensemble (mean-normalized)"
    localization_desc = (
        "full-vocabulary argmax over Trident's own real dense scores (get_trident_seg's DINO+SAM-guided "
        "window fusion, dotted with the full query-feature bank; PAMR off, matching this project's actual "
        "Trident leaderboard config)"
    )
    resize_scale = (2048, 448)

    def __init__(self, spec: DatasetSpec, device: str):
        method_root = ROOT / "third_party" / "official_methods" / "Trident"
        # `segment_anything` must be evicted too: the project's generic SAM
        # AMG proposal probe (_build_sam) already imports the plain pip
        # package earlier in the same process, but Trident vendors its own
        # patched copy (adds `last_attn`/`last_v` attributes to
        # ImageEncoderViT for its window-guidance mechanism) at
        # third_party/official_methods/Trident/segment_anything/ -- without
        # evicting the cached pip module first, `from segment_anything import
        # ...` below would silently reuse the unpatched pip version and
        # `get_sam_feat` would crash on a missing `last_attn` attribute.
        _deregister_mmseg("Trident")
        _evict(["open_clip", "prompts", "trident", "myutils", "seg_utils", "pamr", "custom_datasets", "segment_anything"])
        sys.path.insert(0, str(method_root))
        from trident import Trident

        self.device = device
        name_path = method_root / "configs" / f"cls_{CLS_FILE_KEY[spec.key]}.txt"
        sam_ckpt = ROOT / "weights" / "sam_vit_b_01ec64.pth"
        self.segmentor = Trident(
            clip_type="openai",
            model_type="ViT-B/16",
            vfm_model="dino",
            name_path=str(name_path),
            device=device,
            slide_stride=224,
            slide_crop=336,
            sam_model_type="vit_b",
            sam_ckpt=str(sam_ckpt),
            sam_refinement=False,
            pamr_steps=0,
        )

        # A second, genuinely unmodified CLIP for naming: Trident's own
        # `CLIP.encode_image` requires `external_feats` with no valid None
        # default (its VisionTransformer.forward unconditionally runs
        # custom_attn/guidance_attn, which crash without DINO/SAM tensors),
        # so it cannot serve as a plain global crop encoder. Use the stock
        # pip open_clip captured at module-import time instead, loading the
        # same public OpenAI ViT-B/16 checkpoint Trident itself uses.
        self.naming_net, _, _ = _stock_open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
        self.naming_net.eval().to(device)
        self.naming_tokenizer = _stock_open_clip.get_tokenizer("ViT-B-16-quickgelu")

        import torch
        from prompts.imagenet_template import openai_imagenet_template

        vocabulary = _read_vocab(spec.vocab)
        if len(vocabulary) != spec.num_classes:
            raise ValueError(f"{spec.key}: expected {spec.num_classes} vocab names, got {len(vocabulary)}")
        self.vocabulary = vocabulary
        features = []
        with torch.no_grad():
            for label in vocabulary:
                tokens = self.naming_tokenizer([t(label) for t in openai_imagenet_template]).to(device)
                text = self.naming_net.encode_text(tokens)
                text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                text = text.mean(dim=0)
                text = text / text.norm().clamp_min(1e-6)
                features.append(text)
        self.text_features = torch.stack(features)

        self.query_idx = self.segmentor.query_idx
        self.num_cls = self.segmentor.num_classes
        if self.num_cls != spec.num_classes:
            raise ValueError(f"{spec.key}: Trident query_idx implies {self.num_cls} classes, vocab has {spec.num_classes}")

    def dense_argmax_map(self, image: Image.Image, image_path: str) -> np.ndarray:
        import torch

        tensor = _resize_keep_ratio(image, self.resize_scale).unsqueeze(0).to(self.device)
        img_metas = [{"ori_shape": (image.height, image.width)}]
        with torch.inference_mode():
            logits = self.segmentor.get_trident_seg(tensor, image_path, img_metas)
            if self.segmentor.pamr is not None:
                img_resized = torch.nn.functional.interpolate(
                    tensor, size=(image.height, image.width), mode="bilinear", align_corners=False
                )
                try:
                    logits = self.segmentor.pamr(img_resized, logits.to(img_resized.dtype)).to(logits.dtype)
                except RuntimeError:
                    pass
            probs = (logits[0] * self.segmentor.logit_scale).softmax(0)
            num_queries = probs.shape[0]
            if self.num_cls != num_queries:
                # Mirror Trident's own postprocess_result background-collapse
                # exactly (max over the leading "extra" queries into one
                # background channel), not the scatter_reduce synonym-merge
                # NACLIP/SC-CLIP/CASS use -- Trident's own vocab scheme is
                # different (see trident.py postprocess_result).
                seg_fg = probs[-self.num_cls + 1 :]
                seg_bg = probs[: -self.num_cls + 1].max(0)[0]
                probs = torch.cat([seg_bg.unsqueeze(0), seg_fg], dim=0)
            argmax_map = probs.argmax(dim=0).cpu().numpy()
        return argmax_map

    def naming_predictions(self, image: Image.Image, regions: list[tuple[int, np.ndarray]]) -> list[int]:
        import torch

        crops = [_mask_crop(image, mask) for _, mask in regions]
        preprocess = _clip_preprocess_transform()
        predictions: list[int] = []
        for start in range(0, len(crops), 32):
            batch = torch.stack([preprocess(c) for c in crops[start : start + 32]]).to(self.device)
            with torch.inference_mode():
                feats = self.naming_net.encode_image(batch)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                predictions.extend((feats.float() @ self.text_features.float().T).argmax(dim=1).cpu().tolist())
        return predictions


class CassBackend(MethodBackend):
    name = "cass"
    backbone_desc = (
        "CASS's own real pipeline: OpenAI CLIP ViT-B/16 (its own vendored checkpoint) + DINO ViT-B/8 spectral "
        "fusion (forward_slide, PAMR off per this project's --pamr off leaderboard runs); naming reuses the "
        "same loaded vision tower via encode_image(..., return_cls=True), which bypasses the DINO/spectral fusion"
    )
    prompt_desc = "CASS's own openai_imagenet_template ensemble (mean-normalized)"
    localization_desc = (
        "full-vocabulary argmax over CASS's own real dense scores (forward_slide's hierarchical-prompt "
        "spectral-graph fusion; PAMR off)"
    )
    resize_scale = (2048, 336)

    def __init__(self, spec: DatasetSpec, device: str):
        method_root = ROOT / "third_party" / "official_methods" / "CASS"
        _deregister_mmseg("CASS_segmentor")
        _evict(["clip", "prompts", "cass_segmentor", "custom_datasets", "pamr", "dino", "dinov2"])
        sys.path.insert(0, str(method_root))
        from cass_segmentor import CASS_segmentor

        self.device = device
        name_path = method_root / "configs" / f"cls_{CLS_FILE_KEY[spec.key]}.txt"
        weights = {
            "context459": dict(global_semantics_weight=0.25, mean_vector_weight=0.04, h_threshold=0.09),
            "ade847": dict(global_semantics_weight=0.3, mean_vector_weight=0.05, h_threshold=0.06),
        }[spec.key]
        self.segmentor = CASS_segmentor(
            clip_path="ViT-B/16",
            name_path=str(name_path),
            device=device,
            dino_type="dino_vitb8",
            pamr_steps=0,  # this project's actual CASS leaderboard runs use --pamr off
            dataset=f"./configs/cfg_{spec.key}.py",
            **weights,
        )

        import torch
        from prompts.imagenet_template import openai_imagenet_template

        vocabulary = _read_vocab(spec.vocab)
        if len(vocabulary) != spec.num_classes:
            raise ValueError(f"{spec.key}: expected {spec.num_classes} vocab names, got {len(vocabulary)}")
        self.vocabulary = vocabulary
        # CASS's own encode_image(..., return_cls=True) bypasses the
        # DINO/spectral fusion entirely (confirmed by reading clip/model.py:
        # the last transformer block runs unmodified before that check), so
        # naming reuses self.segmentor.net directly -- no second checkpoint
        # needed, unlike SC-CLIP/Trident.
        import clip as cass_clip  # clip.tokenize is a module-level function, not a model method.

        features = []
        with torch.no_grad():
            for label in vocabulary:
                tokens = cass_clip.tokenize([t(label) for t in openai_imagenet_template]).to(device)
                text = self.segmentor.net.encode_text(tokens)
                text = text / text.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                text = text.mean(dim=0)
                text = text / text.norm().clamp_min(1e-6)
                features.append(text)
        self.text_features = torch.stack(features)

        self.query_idx = self.segmentor.query_idx
        self.num_cls = int(self.query_idx.max().item()) + 1
        if self.num_cls != spec.num_classes:
            raise ValueError(f"{spec.key}: CASS query_idx implies {self.num_cls} classes, vocab has {spec.num_classes}")

    def dense_argmax_map(self, image: Image.Image, image_path: str) -> np.ndarray:
        import torch

        tensor = _resize_keep_ratio(image, self.resize_scale).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.segmentor.forward_slide(tensor, self.segmentor.slide_stride, self.segmentor.slide_crop)
            logits = torch.nn.functional.interpolate(
                logits, size=(image.height, image.width), mode="bilinear", align_corners=self.segmentor.align_corners
            )
            probs = logits[0]
            if self.segmentor.pamr is not None:
                img_resized = torch.nn.functional.interpolate(
                    tensor, size=(image.height, image.width), mode="bilinear", align_corners=self.segmentor.align_corners
                )
                try:
                    probs = self.segmentor.pamr(img_resized, probs.unsqueeze(0).to(img_resized.dtype))[0].to(probs.dtype)
                except RuntimeError:
                    pass
            probs = (probs * self.segmentor.logit_scale).softmax(0)
            if self.num_cls != probs.shape[0]:
                class_probs = probs.new_zeros((self.num_cls, *probs.shape[1:]))
                index = self.query_idx.view(-1, 1, 1).expand_as(probs)
                class_probs.scatter_reduce_(0, index, probs, reduce="amax", include_self=True)
                probs = class_probs
            argmax_map = probs.argmax(dim=0).cpu().numpy()
        return argmax_map

    def naming_predictions(self, image: Image.Image, regions: list[tuple[int, np.ndarray]]) -> list[int]:
        import torch

        preprocess = _clip_preprocess_transform()
        predictions: list[int] = []
        for class_id, mask in regions:
            crop = _mask_crop(image, mask)
            crop_tensor = preprocess(crop).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                feats = self.segmentor.net.encode_image(
                    crop_tensor, crop, self.segmentor.dino_type, self.segmentor.dino_model, self.segmentor.dataset,
                    return_all=False, return_cls=True,
                )
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                pred = int((feats.float() @ self.text_features.float().T).argmax(dim=1).item())
            predictions.append(pred)
        return predictions


class SamAmgSiglipBackend(MethodBackend):
    name = "sam_amg_siglip"
    backbone_desc = (
        "Real SAM ViT-H automatic mask generation + SigLIP (ViT-SO400M-14-SigLIP, webli) naming, "
        "exactly mirroring src/tf_ovos/adapters/sam_amg_siglip.py::SamAmgSiglipAdapter"
    )
    prompt_desc = "SamAmgSiglipAdapter's own single template: 'a photo of a {}' (no multi-template ensemble)"
    localization_desc = (
        "Not a dense per-pixel score tensor -- proposal+naming architectures have no continuous competing-class "
        "map (see generate_masks-family methods for that). Reuses this method's own real final compositing rule "
        "(SamAmgSiglipAdapter._predict_one): every real SAM proposal is SigLIP-scored via argmax over the full "
        "vocabulary, then painted onto the pixel grid largest-area-first, gated by predicted_iou*stability_score. "
        "The resulting genuine per-pixel label map is scored the same way generate_diagnostic_tables.py already "
        "defines gt_text_localization_iou for any method, computed here from live SAM+SigLIP calls rather than a "
        "saved artifact."
    )

    def __init__(self, spec: DatasetSpec, device: str):
        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        self.torch = torch
        self.device = device
        sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT))
        sam.to(device).eval()
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=400,
        )
        self.siglip_model, self.siglip_preprocess, _ = _stock_open_clip.create_model_and_transforms(
            "ViT-SO400M-14-SigLIP", pretrained="webli"
        )
        self.siglip_model.eval().to(device)
        self.tokenizer = _stock_open_clip.get_tokenizer("ViT-SO400M-14-SigLIP")

        vocabulary = _read_vocab(spec.vocab)
        if len(vocabulary) != spec.num_classes:
            raise ValueError(f"{spec.key}: expected {spec.num_classes} vocab names, got {len(vocabulary)}")
        self.vocabulary = vocabulary
        prompts = [f"a photo of a {c}" for c in vocabulary]
        tokens = self.tokenizer(prompts).to(device)
        with torch.inference_mode():
            feats = self.siglip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.text_features = feats

    def _encode_crops(self, crops: list[Image.Image]):
        torch = self.torch
        if not crops:
            return torch.zeros((0, self.text_features.shape[1]), device=self.device)
        tensors = torch.stack([self.siglip_preprocess(c) for c in crops]).to(self.device)
        with torch.inference_mode():
            feats = self.siglip_model.encode_image(tensors)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return feats

    def dense_argmax_map(self, image: Image.Image, image_path: str) -> np.ndarray:
        width, height = image.size
        masks = self.mask_generator.generate(np.asarray(image))
        if not masks:
            crop_feat = self._encode_crops([image])
            sims = (crop_feat @ self.text_features.T)[0].float().cpu().numpy()
            return np.full((height, width), int(np.argmax(sims)), dtype=np.int64)

        masks_sorted = sorted(masks, key=lambda m: m["area"], reverse=True)
        crops = []
        for m in masks_sorted:
            x, y, w, h = [int(v) for v in m["bbox"]]
            x2, y2 = min(x + w, width), min(y + h, height)
            crop = image.crop((x, y, x2, y2))
            if crop.size[0] < 1 or crop.size[1] < 1:
                crop = image
            crops.append(crop)
        crop_features = self._encode_crops(crops)
        sims = (crop_features @ self.text_features.T).float().cpu().numpy()
        best_class = np.argmax(sims, axis=1)

        label_map = np.zeros((height, width), dtype=np.int64)
        confidence_map = np.full((height, width), -1.0, dtype=np.float32)
        for i, m in enumerate(masks_sorted):
            seg = m["segmentation"]
            conf = float(m["predicted_iou"]) * float(m["stability_score"])
            cls = int(best_class[i])
            update = seg & (conf > confidence_map)
            label_map[update] = cls
            confidence_map[update] = conf
        return label_map

    def naming_predictions(self, image: Image.Image, regions: list[tuple[int, np.ndarray]]) -> list[int]:
        crops = [_mask_crop(image, mask) for _, mask in regions]
        feats = self._encode_crops(crops)
        if feats.shape[0] == 0:
            return []
        sims = (feats @ self.text_features.T).float().cpu().numpy()
        return [int(v) for v in np.argmax(sims, axis=1)]


_PREPROCESS_CACHE: Any = None


def _clip_preprocess_transform():
    """Standard OpenAI-CLIP preprocessing (resize 224, center crop, CLIP
    normalize), identical regardless of which vendored ``clip`` copy built
    it — this is deliberately reconstructed by hand (not re-imported from a
    per-method vendored module) so it stays stable across backend switches
    within one process."""
    global _PREPROCESS_CACHE
    if _PREPROCESS_CACHE is None:
        from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, Resize, ToTensor

        _PREPROCESS_CACHE = Compose(
            [
                Resize(224, interpolation=InterpolationMode.BICUBIC),
                CenterCrop(224),
                lambda img: img.convert("RGB"),
                ToTensor(),
                Normalize(CLIP_MEAN, CLIP_STD),
            ]
        )
    return _PREPROCESS_CACHE


def _read_vocab(path: Path) -> list[str]:
    sys.path.insert(0, str(ROOT / "src"))
    from tf_ovos.data import read_vocab

    return read_vocab(path)


BACKENDS: dict[str, Callable[[DatasetSpec, str], MethodBackend]] = {
    "naclip": NaclipBackend,
    "scclip": ScclipBackend,
    "trident": TridentBackend,
    "cass": CassBackend,
    "sam_amg_siglip": SamAmgSiglipBackend,
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
    sam_empty_images: int = 0
    sam_proposals_total: int = 0

    def note_error(self, stage: str, sample_id: str, exc: BaseException) -> None:
        key = f"{stage}:{type(exc).__name__}"
        self.errors[key] += 1
        if len(self.error_examples) < 8:
            self.error_examples.append(f"{sample_id} [{stage}]: {type(exc).__name__}: {exc}")

    def row(self, method: str, backend: MethodBackend) -> dict[str, Any]:
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
            "backbone": backend.backbone_desc,
            "prompt_ensemble": backend.prompt_desc,
            "localization_binarization": backend.localization_desc,
            "sam_checkpoint": str(SAM_CHECKPOINT),
            "sam_parameters": "points_per_side=32;pred_iou_thresh=0.86;stability_score_thresh=0.92;min_mask_region_area=400",
            "error_counts": json.dumps(dict(self.errors), sort_keys=True),
            "error_examples": " | ".join(self.error_examples),
        }


def run_dataset(spec: DatasetSpec, backend: MethodBackend, sam_generator, args: argparse.Namespace) -> DiagnosticAccumulator:
    from tf_ovos.data import load_manifest
    from tf_ovos.metrics import iou, load_label_map

    samples = load_manifest(spec.manifest)[: args.num_images]
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
            regions = [(class_id, gt == class_id) for class_id in class_ids]
            if not regions:
                raise ValueError("no valid GT semantic classes")

            predicted_ids = backend.naming_predictions(image, regions)
            acc.naming_regions += len(regions)
            acc.naming_hits += sum(int(pred == class_id) for pred, (class_id, _) in zip(predicted_ids, regions))

            argmax_map = backend.dense_argmax_map(image, str(sample.image_path))
            for class_id, gt_mask in regions:
                predicted_mask = argmax_map == class_id
                acc.localization_iou_sum += iou(predicted_mask, gt_mask)
                acc.localization_regions += 1

            sam_masks = sam_generator.generate(np.asarray(image))
            acc.sam_proposals_total += len(sam_masks)
            if not sam_masks:
                acc.sam_empty_images += 1
            best_ious = _sam_best_ious(sam_masks, regions)
            acc.proposal_regions += len(best_ious)
            acc.proposal_iou_sum += float(sum(best_ious))
            acc.proposal_recall_hits += sum(v >= 0.5 for v in best_ious)
            acc.processed_images += 1
        except Exception as exc:  # keep the pilot auditable instead of fabricating a result
            acc.note_error("image", sample.image_id, exc)
            print(f"[{spec.key} {index}/{len(samples)}] skipped {sample.image_id}: {type(exc).__name__}: {exc}", flush=True)
            import torch

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
    keys = ["gt_region_naming_top1", "gt_text_localization_iou", "proposal_oracle_iou", "proposal_recall_at_05"]
    row: dict[str, Any] = {
        "row_type": "unweighted_dataset_mean",
        "dataset": "Context-459 + ADE-847 mean",
        "dataset_key": "mean",
        "method": rows[0]["method"],
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(BACKENDS))
    parser.add_argument("--datasets", nargs="+", choices=sorted(SPECS), default=["context459", "ade847"])
    parser.add_argument("--num-images", type=int, default=250)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--progress-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
    if args.output_csv is None:
        args.output_csv = ROOT / "runs" / "analysis" / f"obs5_real_diagnostic_{args.method}.csv"

    sam_generator = _build_sam(args.device)
    rows = []
    accs = []
    for key in args.datasets:
        spec = SPECS[key]
        backend_cls = BACKENDS[args.method]
        backend = backend_cls(spec, args.device)
        acc = run_dataset(spec, backend, sam_generator, args)
        accs.append(acc)
        rows.append(acc.row(args.method, backend))
        del backend
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
