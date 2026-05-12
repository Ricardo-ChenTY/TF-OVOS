import json
from pathlib import Path

from PIL import Image

from tf_ovcos.make_manifest import build_manifest
from tf_ovcos.make_shards import make_shards
from tf_ovcos.merge_predictions import merge_predictions
from tf_ovcos.run_method import run_method


def write_mask(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (4, 4), value).save(path)


def test_manifest_shard_run_merge_eval_paths(tmp_path):
    image_dir = tmp_path / "raw" / "images"
    mask_dir = tmp_path / "raw" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for image_id in ["a", "b", "c"]:
        Image.new("RGB", (4, 4), (255, 0, 0)).save(image_dir / f"{image_id}.jpg")
        write_mask(mask_dir / f"{image_id}.png", 255)

    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({"a": "frog", "b": "fish", "c": "bird"}), encoding="utf-8")
    manifest = tmp_path / "data" / "manifests" / "toy.jsonl"
    rows = build_manifest(image_dir, mask_dir, manifest, labels_path, False, True, "image_id", "label")
    assert len(rows) == 3
    from tf_ovcos.data import write_jsonl

    write_jsonl(rows, manifest)

    shard_dir = tmp_path / "data" / "manifests" / "shards" / "toy"
    shard_paths = make_shards(manifest, 2, shard_dir, "round-robin")
    assert len(shard_paths) == 2

    for shard_path in shard_paths:
        out_dir = tmp_path / "runs" / "debug_copy_gt" / shard_path.stem
        run_method("debug_copy_gt", shard_path, None, out_dir, False)
        target_dir = tmp_path / "runs" / "debug_copy_gt" / "toy" / "shards" / shard_path.stem
        target_dir.mkdir(parents=True)
        (out_dir / "predictions.jsonl").replace(target_dir / "predictions.jsonl")
        pred_masks = target_dir / "pred_masks"
        pred_masks.mkdir()
        for mask in (out_dir / "pred_masks").glob("*.png"):
            mask.replace(pred_masks / mask.name)

    merged = merge_predictions(
        manifest,
        tmp_path / "runs" / "debug_copy_gt" / "toy" / "shards",
        tmp_path / "runs" / "debug_copy_gt" / "toy" / "predictions.jsonl",
    )
    assert [row["image_id"] for row in merged] == ["a", "b", "c"]
