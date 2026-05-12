from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


EPS = 1e-8


def load_binary_mask(path: str, threshold: float = 0.5) -> np.ndarray:
    image = Image.open(path).convert("L")
    arr = np.asarray(image, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return arr >= threshold


def resize_mask_to(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if mask.shape == shape:
        return mask
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, gt).sum() / union)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean())


def f_beta(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    precision = tp / max(pred.sum(), EPS)
    recall = tp / max(gt.sum(), EPS)
    return float((1 + beta2) * precision * recall / max(beta2 * precision + recall, EPS))


def e_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_f = pred.astype(np.float32)
    gt_f = gt.astype(np.float32)
    pred_centered = pred_f - pred_f.mean()
    gt_centered = gt_f - gt_f.mean()
    align = 2 * pred_centered * gt_centered / (pred_centered**2 + gt_centered**2 + EPS)
    enhanced = ((align + 1) ** 2) / 4
    return float(enhanced.mean())


def boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    dilated = np.asarray(image.filter(ImageFilter.MaxFilter(radius * 2 + 1))) > 0
    eroded = np.asarray(image.filter(ImageFilter.MinFilter(radius * 2 + 1))) > 0
    return np.logical_xor(dilated, eroded)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, radius: int = 2) -> float:
    return iou(boundary(pred, radius), boundary(gt, radius))


@dataclass(frozen=True)
class MaskMetrics:
    iou: float
    f_beta: float
    e_measure: float
    mae: float
    boundary_iou: float


def compute_mask_metrics(pred: np.ndarray, gt: np.ndarray) -> MaskMetrics:
    pred = resize_mask_to(pred, gt.shape)
    return MaskMetrics(
        iou=iou(pred, gt),
        f_beta=f_beta(pred, gt),
        e_measure=e_measure(pred, gt),
        mae=mae(pred, gt),
        boundary_iou=boundary_iou(pred, gt),
    )


def class_aware(metrics: MaskMetrics, pred_label: str | None, gt_label: str | None) -> dict[str, float]:
    exact = pred_label == gt_label and gt_label is not None
    gate = 1.0 if exact else 0.0
    return {
        "cIoU": metrics.iou * gate,
        "cF_beta": metrics.f_beta * gate,
        "cE_m": metrics.e_measure * gate,
        "cBIoU": metrics.boundary_iou * gate,
        "cMAE": metrics.mae if exact else 1.0,
        "Exact": gate,
    }


def ambiguity_rows(rows: list[tuple[float, str | None, str | None]], loc_threshold: float = 0.5) -> dict[str, object]:
    localized = [(gt, pred) for loc_iou, gt, pred in rows if loc_iou >= loc_threshold]
    exact = [(gt, pred) for gt, pred in localized if gt == pred and gt is not None]
    confusions = Counter((gt, pred) for gt, pred in localized if gt != pred and gt is not None and pred is not None)
    loc_at = len(localized) / max(len(rows), 1)
    exact_at = len(exact) / max(len(rows), 1)
    return {
        "Loc@0.5": loc_at,
        "Exact@0.5": exact_at,
        "ClsErr@Loc": 1.0 - exact_at / max(loc_at, EPS),
        "num_edges": len(confusions),
        "top_confusions": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in confusions.most_common(20)
        ],
    }
