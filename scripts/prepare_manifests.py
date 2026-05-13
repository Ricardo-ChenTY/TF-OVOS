"""
Generate JSONL manifests for all benchmark datasets.

Run after downloading raw data with the dl_*.sh scripts.

Usage:
    python scripts/prepare_manifests.py --dataset voc20
    python scripts/prepare_manifests.py --dataset context59
    python scripts/prepare_manifests.py --dataset context459
    python scripts/prepare_manifests.py --dataset ade20k150
    python scripts/prepare_manifests.py --dataset ade20k847
    python scripts/prepare_manifests.py --dataset coco_stuff171
    python scripts/prepare_manifests.py --dataset all   # run all available

Each dataset handler:
  1. Reads the raw annotations
  2. Remaps pixel labels to 0-indexed classes with 255=void
  3. Saves remapped PNGs to data/raw/<dataset>_labels/
  4. Writes a JSONL manifest to data/manifests/<dataset>_val.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
MANIFESTS.mkdir(parents=True, exist_ok=True)


def _save_label_map(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"  wrote {len(rows)} rows → {path.relative_to(ROOT)}")


def _rel(path: Path, base: Path) -> str:
    """Return path relative to base, using ../ notation."""
    return os.path.relpath(str(path), str(base))


# ── VOC 2012 → voc20_val ─────────────────────────────────────────────────────

def prepare_voc20() -> None:
    """
    VOC 2012 segmentation val set.
    Raw label maps: pixel 0=background, 1-20=classes, 255=boundary/void.
    Remap: 1-20 → 0-19,  {0, 255} → 255.
    """
    voc = RAW / "VOCdevkit" / "VOC2012"
    val_list = voc / "ImageSets" / "Segmentation" / "val.txt"
    img_dir = voc / "JPEGImages"
    seg_dir = voc / "SegmentationClass"
    out_dir = RAW / "voc20_labels"

    if not val_list.exists():
        raise FileNotFoundError(f"VOC 2012 not found at {voc}. Run: bash scripts/dl_voc20.sh")

    image_ids = val_list.read_text().strip().splitlines()
    rows = []
    for image_id in image_ids:
        src = seg_dir / f"{image_id}.png"
        if not src.exists():
            continue  # no segmentation annotation for this image
        raw = np.array(Image.open(src))
        label = np.full_like(raw, 255, dtype=np.uint8)
        # classes 1-20 → 0-19
        mask = (raw >= 1) & (raw <= 20)
        label[mask] = raw[mask] - 1
        dst = out_dir / f"{image_id}.png"
        _save_label_map(label, dst)
        rows.append({
            "image_id": image_id,
            "image_path": _rel(img_dir / f"{image_id}.jpg", MANIFESTS),
            "mask_path": _rel(dst, MANIFESTS),
        })

    _write_jsonl(rows, MANIFESTS / "voc20_val.jsonl")
    print(f"  label maps → {out_dir.relative_to(ROOT)}")


# ── PASCAL Context → context59_val / context459_val ──────────────────────────

def prepare_context(num_classes: int) -> None:
    """
    PASCAL Context 59 or 459.
    Requires: pip install detail
    Raw annotations: trainval.json + VOC 2010 JPEGImages.
    Label map convention: class 0 = background (void), 1-N = N classes.
    Remap: 1-N → 0-(N-1),  0 → 255.
    """
    try:
        from detail import Detail  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "The 'detail' package is required for PASCAL Context.\n"
            "Install with:  pip install detail"
        )

    ctx_dir = RAW / "pascal_context"
    img_dir = RAW / "VOCdevkit" / "VOC2010" / "JPEGImages"
    annotation_file = ctx_dir / "trainval.json"

    if not annotation_file.exists():
        raise FileNotFoundError(
            f"Annotations not found at {annotation_file}. Run: bash scripts/dl_context.sh"
        )
    if not img_dir.exists():
        raise FileNotFoundError(
            f"VOC 2010 images not found at {img_dir}. Run: bash scripts/dl_context.sh"
        )

    tag = f"ctx{num_classes}"
    out_dir = RAW / f"context{num_classes}_labels"
    manifest_path = MANIFESTS / f"context{num_classes}_val.jsonl"

    d = Detail(str(annotation_file), str(img_dir), num_classes)
    image_ids = d.getImgs(phase="val")

    rows = []
    for info in image_ids:
        img_id = info["id"]
        filename = info["file_name"]  # e.g. "2008_000015.jpg"
        stem = Path(filename).stem

        label_map = d.getMask(info, dtype="numpy")  # H×W, values 0..num_classes
        label = np.full_like(label_map, 255, dtype=np.uint8)
        mask = (label_map >= 1) & (label_map <= num_classes)
        label[mask] = label_map[mask] - 1

        dst = out_dir / f"{stem}.png"
        _save_label_map(label, dst)
        rows.append({
            "image_id": stem,
            "image_path": _rel(img_dir / filename, MANIFESTS),
            "mask_path": _rel(dst, MANIFESTS),
        })

    _write_jsonl(rows, manifest_path)
    print(f"  label maps → {out_dir.relative_to(ROOT)}")


# ── ADE20K-150 → ade20k150_val ───────────────────────────────────────────────

def prepare_ade20k150() -> None:
    """
    ADE20K-150 (MIT ADEChallengeData2016) validation set.
    Raw label maps: pixel 0=void, 1-150=classes.
    Remap: 1-150 → 0-149,  0 → 255.
    """
    ade = RAW / "ADEChallengeData2016"
    img_dir = ade / "images" / "validation"
    seg_dir = ade / "annotations" / "validation"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"ADE20K-150 not found at {ade}. Run: bash scripts/dl_ade20k.sh"
        )

    out_dir = RAW / "ade20k150_labels"
    rows = []
    for seg_path in sorted(seg_dir.glob("*.png")):
        stem = seg_path.stem
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            continue
        raw = np.array(Image.open(seg_path))
        label = np.full_like(raw, 255, dtype=np.uint8)
        mask = (raw >= 1) & (raw <= 150)
        label[mask] = raw[mask] - 1
        dst = out_dir / f"{stem}.png"
        _save_label_map(label, dst)
        rows.append({
            "image_id": stem,
            "image_path": _rel(img_path, MANIFESTS),
            "mask_path": _rel(dst, MANIFESTS),
        })

    _write_jsonl(rows, MANIFESTS / "ade20k150_val.jsonl")
    print(f"  label maps → {out_dir.relative_to(ROOT)}")


# ── ADE20K-847 → ade20k847_val ───────────────────────────────────────────────

def prepare_ade20k847() -> None:
    """
    Full ADE20K (847 classes) validation split.
    The 2021 release uses per-scene annotation PNGs with the same convention
    as ADE20K-150: pixel 0=void, 1-N=class index.

    Expected layout after extracting ADE20K_2021_17_01.zip:
      data/raw/ADE20K_2021_17_01/images/ADE/validation/<scene>/<scene>_<n>.jpg
      data/raw/ADE20K_2021_17_01/images/ADE/validation/<scene>/<scene>_<n>_seg.png
    """
    ade847 = RAW / "ADE20K_2021_17_01"
    val_root = ade847 / "images" / "ADE" / "validation"

    if not val_root.exists():
        raise FileNotFoundError(
            f"ADE20K-847 not found at {ade847}.\n"
            "Manual download required:\n"
            "  1. Go to http://groups.csail.mit.edu/vision/datasets/ADE20K/request_data.php\n"
            "  2. Download ADE20K_2021_17_01.zip\n"
            f"  3. Extract to {RAW}"
        )

    out_dir = RAW / "ade20k847_labels"
    rows = []
    for seg_path in sorted(val_root.rglob("*_seg.png")):
        stem = seg_path.name.replace("_seg.png", "")
        img_path = seg_path.parent / f"{stem}.jpg"
        if not img_path.exists():
            continue
        raw = np.array(Image.open(seg_path))
        # ADE20K seg PNGs encode class as (R/10)*256 + G; R channel = class//10, G = class%10
        # In the 2021 release, the seg PNG is a paletted or RGB file.
        if raw.ndim == 3:
            raw_class = (raw[:, :, 0].astype(np.uint16) // 10) * 256 + raw[:, :, 1]
        else:
            raw_class = raw.astype(np.uint16)
        label = np.full(raw_class.shape, 255, dtype=np.uint8)
        mask = (raw_class >= 1) & (raw_class <= 847)
        label[mask] = (raw_class[mask] - 1).astype(np.uint8)
        dst = out_dir / f"{stem}.png"
        _save_label_map(label, dst)
        rows.append({
            "image_id": stem,
            "image_path": _rel(img_path, MANIFESTS),
            "mask_path": _rel(dst, MANIFESTS),
        })

    _write_jsonl(rows, MANIFESTS / "ade20k847_val.jsonl")
    print(f"  label maps → {out_dir.relative_to(ROOT)}")


# ── COCO-Stuff 171 → coco_stuff171_val ───────────────────────────────────────

# COCO category IDs in our vocab order (things 0-79, then stuff 80-170).
# Things: 80 categories (COCO IDs skip a few numbers due to retired classes).
_COCO_THING_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]
# Stuff: 91 categories with contiguous IDs 92-182.
_COCO_STUFF_IDS = list(range(92, 183))

# Build lookup table: COCO category ID → our vocab index (0-170)
_COCO_ID_TO_VOCAB: dict[int, int] = {}
for _i, _cid in enumerate(_COCO_THING_IDS):
    _COCO_ID_TO_VOCAB[_cid] = _i
for _i, _cid in enumerate(_COCO_STUFF_IDS):
    _COCO_ID_TO_VOCAB[_cid] = 80 + _i


def prepare_coco_stuff171() -> None:
    """
    COCO-Stuff 171 validation set.
    stuffthingmaps PNG: pixel = COCO category ID (1-182), 0 = unlabeled.
    Remap via _COCO_ID_TO_VOCAB lookup table.
    """
    coco = RAW / "coco_stuff171"
    img_dir = coco / "val2017"
    ann_dir = coco / "annotations" / "val2017"

    if not img_dir.exists():
        raise FileNotFoundError(
            f"COCO val2017 images not found at {img_dir}. Run: bash scripts/dl_coco_stuff.sh"
        )
    if not ann_dir.exists():
        raise FileNotFoundError(
            f"COCO-Stuff annotations not found at {ann_dir}. Run: bash scripts/dl_coco_stuff.sh"
        )

    # Build numpy lookup table for fast vectorized remap (size 256)
    lut = np.full(256, 255, dtype=np.uint8)
    for coco_id, vocab_idx in _COCO_ID_TO_VOCAB.items():
        if coco_id < 256:
            lut[coco_id] = vocab_idx

    out_dir = RAW / "coco_stuff171_labels"
    rows = []
    for ann_path in sorted(ann_dir.glob("*.png")):
        image_id = ann_path.stem  # e.g. "000000000139"
        img_path = img_dir / f"{image_id}.jpg"
        if not img_path.exists():
            continue
        raw = np.array(Image.open(ann_path))
        label = lut[raw]
        dst = out_dir / f"{image_id}.png"
        _save_label_map(label, dst)
        rows.append({
            "image_id": image_id,
            "image_path": _rel(img_path, MANIFESTS),
            "mask_path": _rel(dst, MANIFESTS),
        })

    _write_jsonl(rows, MANIFESTS / "coco_stuff171_val.jsonl")
    print(f"  label maps → {out_dir.relative_to(ROOT)}")


# ── CLI ───────────────────────────────────────────────────────────────────────

HANDLERS = {
    "voc20": prepare_voc20,
    "context59": lambda: prepare_context(59),
    "context459": lambda: prepare_context(459),
    "ade20k150": prepare_ade20k150,
    "ade20k847": prepare_ade20k847,
    "coco_stuff171": prepare_coco_stuff171,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare benchmark manifests.")
    parser.add_argument(
        "--dataset",
        choices=[*HANDLERS, "all"],
        required=True,
        help="Which dataset to prepare (or 'all' to attempt every dataset).",
    )
    args = parser.parse_args()

    targets = list(HANDLERS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        print(f"\n[{name}] preparing...")
        try:
            HANDLERS[name]()
            print(f"[{name}] OK")
        except FileNotFoundError as e:
            print(f"[{name}] SKIP — {e}")
        except ImportError as e:
            print(f"[{name}] SKIP — {e}")


if __name__ == "__main__":
    main()
