from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tf_ovcos.data import iter_missing_predictions, load_manifest, load_predictions
from tf_ovcos.metrics import ambiguity_rows, class_aware, compute_mask_metrics, load_binary_mask


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def evaluate(manifest: Path, predictions_path: Path, task: str, threshold: float) -> dict[str, object]:
    samples = load_manifest(manifest)
    predictions = load_predictions(predictions_path)
    missing = iter_missing_predictions(samples, predictions)
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing {len(missing)} predictions. First missing ids: {preview}")

    mask_rows: list[dict[str, float]] = []
    class_rows: list[dict[str, float]] = []
    ambiguity_input: list[tuple[float, str | None, str | None]] = []

    for sample in samples:
        pred = predictions[sample.image_id]
        gt_mask = load_binary_mask(str(sample.mask_path), threshold=threshold)
        pred_mask = load_binary_mask(str(pred.mask_path), threshold=threshold)
        metrics = compute_mask_metrics(pred_mask, gt_mask)
        mask_rows.append(
            {
                "IoU": metrics.iou,
                "F_beta": metrics.f_beta,
                "E_m": metrics.e_measure,
                "MAE": metrics.mae,
                "BIoU": metrics.boundary_iou,
            }
        )
        ambiguity_input.append((metrics.iou, sample.label, pred.label))
        if task == "class-aware":
            class_rows.append(class_aware(metrics, pred.label, sample.label))

    result: dict[str, object] = {
        "num_samples": len(samples),
        "task": task,
        "mask_metrics": mean_dict(mask_rows),
    }
    if task == "class-aware":
        result["class_aware_metrics"] = mean_dict(class_rows)
        result["ambiguity"] = ambiguity_rows(ambiguity_input)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TF-OVCOS predictions.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--task", choices=["class-aware", "mask-only"], required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    result = evaluate(args.manifest, args.predictions, args.task, args.threshold)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
