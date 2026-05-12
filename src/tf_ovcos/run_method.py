from __future__ import annotations

import argparse
from pathlib import Path

from tf_ovcos.adapters.base import write_predictions
from tf_ovcos.adapters.registry import ADAPTERS, adapter_info
from tf_ovcos.data import load_manifest, read_vocab


def run_method(method: str, manifest: Path, vocab: Path | None, out_dir: Path, skip_existing: bool) -> Path:
    if method not in ADAPTERS:
        available = ", ".join(sorted(ADAPTERS))
        raise ValueError(f"Unknown method {method!r}. Available adapters: {available}")

    adapter_cls = ADAPTERS[method]
    if not getattr(adapter_cls, "runnable", True):
        hint = getattr(adapter_cls, "setup_hint", "")
        raise NotImplementedError(f"Adapter {method!r} is not implemented yet. {hint}")

    predictions_path = out_dir / "predictions.jsonl"
    if skip_existing and predictions_path.exists():
        print(f"Skipping existing predictions: {predictions_path}")
        return predictions_path

    samples = load_manifest(manifest)
    vocabulary = read_vocab(vocab) if vocab else []
    adapter = adapter_cls()
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
        for info in adapter_info():
            status = "runnable" if info.runnable else "planned"
            print(f"{info.name}\t{status}")
        return

    if not args.method or not args.manifest or not args.out:
        parser.error("--method, --manifest, and --out are required unless --list-methods is used")
    run_method(args.method, args.manifest, args.vocab, args.out, args.skip_existing)


if __name__ == "__main__":
    main()
