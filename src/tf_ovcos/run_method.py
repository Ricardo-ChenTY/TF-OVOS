from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from tf_ovcos.adapters.base import MethodAdapter, write_predictions
from tf_ovcos.data import Prediction, Sample, as_output_path, load_manifest, read_vocab


class CopyGroundTruthAdapter(MethodAdapter):
    name = "debug_copy_gt"

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        pred_dir = output_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_mask = pred_dir / f"{sample.image_id}.png"
        shutil.copyfile(sample.mask_path, out_mask)
        return Prediction(
            image_id=sample.image_id,
            mask_path=Path(as_output_path(out_mask, output_dir)),
            label=sample.label,
            score=1.0,
            metadata={"debug": "copied ground-truth mask; not a benchmark method"},
        )


class EmptyMaskAdapter(MethodAdapter):
    name = "debug_empty"

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        pred_dir = output_dir / "pred_masks"
        pred_dir.mkdir(parents=True, exist_ok=True)
        out_mask = pred_dir / f"{sample.image_id}.png"
        with Image.open(sample.mask_path) as gt:
            Image.new("L", gt.size, 0).save(out_mask)
        label = vocabulary[0] if vocabulary else sample.label
        return Prediction(
            image_id=sample.image_id,
            mask_path=Path(as_output_path(out_mask, output_dir)),
            label=label,
            score=0.0,
            metadata={"debug": "empty mask; not a benchmark method"},
        )


ADAPTERS: dict[str, type[MethodAdapter]] = {
    CopyGroundTruthAdapter.name: CopyGroundTruthAdapter,
    EmptyMaskAdapter.name: EmptyMaskAdapter,
}


def run_method(method: str, manifest: Path, vocab: Path | None, out_dir: Path, skip_existing: bool) -> Path:
    if method not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown method {method!r}. Available adapters: {available}")

    predictions_path = out_dir / "predictions.jsonl"
    if skip_existing and predictions_path.exists():
        print(f"Skipping existing predictions: {predictions_path}")
        return predictions_path

    samples = load_manifest(manifest)
    vocabulary = read_vocab(vocab) if vocab else []
    adapter = ADAPTERS[method]()
    predictions = adapter.predict_many(samples, vocabulary, out_dir)
    write_predictions(predictions, predictions_path)
    print(f"Wrote {len(predictions)} predictions to {predictions_path}")
    return predictions_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a TF-OVCOS method adapter.")
    parser.add_argument("--method")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list-methods", action="store_true")
    args = parser.parse_args()

    if args.list_methods:
        for name in sorted(ADAPTERS):
            print(name)
        return

    if not args.method or not args.manifest or not args.out:
        parser.error("--method, --manifest, and --out are required unless --list-methods is used")
    run_method(args.method, args.manifest, args.vocab, args.out, args.skip_existing)


if __name__ == "__main__":
    main()
