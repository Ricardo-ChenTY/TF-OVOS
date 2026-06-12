from __future__ import annotations

import io
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
ROWS = [json.loads(line) for line in Path("/private/tmp/ade_hf_rows.jsonl").read_text().splitlines() if line.strip()]
ROW_BY_FILE = {row["row"]["filename"]: row["row"] for row in ROWS}
ROW_IDX_BY_FILE = {row["row"]["filename"]: row["row_idx"] for row in ROWS}
PRED_ROOT = ROOT / "tf_ovcos_full_results_20260526/runs/artifacts/official_predictions"
OUT = ROOT / "TMLR26_Bench/figures"

METHODS = [
    ("CorrCLIP", "corrclip"),
    ("NACLIP", "naclip"),
    ("ProxyCLIP", "proxyclip"),
    ("SCLIP", "sclip"),
]


CASES = {
    "overview_window_ade150": {
        "dataset": "ade20k",
        "vocab": "configs/vocab/ade20k_150.txt",
        "filename": "ADE_val_00001394.jpg",
        "raw_name": "window",
        "target": "windowpane",
        "title": "window",
    },
    "overview_pot_ade150": {
        "dataset": "ade20k",
        "vocab": "configs/vocab/ade20k_150.txt",
        "filename": "ADE_val_00000460.jpg",
        "raw_name": "pot",
        "target": "pot",
        "title": "pot",
    },
    "overview_people_ade150": {
        "dataset": "ade20k",
        "vocab": "configs/vocab/ade20k_150.txt",
        "filename": "ADE_val_00000998.jpg",
        "raw_name": "person",
        "target": "people",
        "title": "people",
    },
    "overview_computer_ade150": {
        "dataset": "ade20k",
        "vocab": "configs/vocab/ade20k_150.txt",
        "filename": "ADE_val_00001724.jpg",
        "raw_name": "computer",
        "target": "computer",
        "title": "computer",
    },
    "case_study_mismatch_box_bookcase": {
        "dataset": "ade20k",
        "vocab": "configs/vocab/ade20k_150.txt",
        "filename": "ADE_val_00000029.jpg",
        "raw_name": "boxes",
        "target": "box",
        "title": "ADE-150 / box",
    },
    "case_study_small_object_bus_independent": {
        "dataset": "ade20k",
        "vocab": "configs/vocab/ade20k_150.txt",
        "filename": "ADE_val_00000735.jpg",
        "raw_name": "bus",
        "target": "bus",
        "title": "ADE-150 / bus",
    },
    "case_study_large_vocab_shelf_ade847": {
        "dataset": "ade847",
        "vocab": "configs/vocab/ade20k_847.txt",
        "filename": "ADE_val_00000456.jpg",
        "raw_name": "shelves",
        "target": "shelf",
        "title": "ADE-847 / shelf",
    },
    "case_study_large_vocab_jacket_ade847": {
        "dataset": "ade847",
        "vocab": "configs/vocab/ade20k_847.txt",
        "filename": "ADE_val_00000255.jpg",
        "raw_name": "jacket",
        "target": "jacket",
        "title": "ADE-847 / jacket",
    },
}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


HEADER_FONT = load_font(28)
LABEL_FONT = load_font(21)
SIDE_FONT = load_font(24)
PANEL_FONT = load_font(22)


def load_vocab(path: str) -> list[str]:
    return [line.strip() for line in (ROOT / path).read_text().splitlines() if line.strip()]


def download_image(url: str) -> Image.Image:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def fresh_row(filename: str) -> dict:
    row_idx = ROW_IDX_BY_FILE[filename]
    url = (
        "https://datasets-server.huggingface.co/rows"
        f"?dataset=uva-cv-lab/ADE20k-150&config=default&split=validation&offset={row_idx}&length=1"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=45) as response:
        data = json.load(response)
    return data["rows"][0]["row"]


def find_object(row: dict, raw_name: str) -> dict:
    raw_key = norm(raw_name)
    candidates = [obj for obj in row["objects"] if norm(str(obj.get("raw_name", ""))) == raw_key]
    if not candidates:
        candidates = [obj for obj in row["objects"] if raw_key in norm(str(obj.get("name", "")))]
    if not candidates:
        raise RuntimeError(f"Could not find object {raw_name} in {row['filename']}")
    return max(candidates, key=lambda obj: polygon_area(obj, row["image"]["width"], row["image"]["height"]))


def polygon_area(obj: dict, width: int, height: int) -> int:
    mask = object_mask(obj, width, height)
    return int(mask.sum())


def object_mask(obj: dict, width: int, height: int) -> np.ndarray:
    poly = obj["polygon"]
    points = list(zip(poly["x"], poly["y"]))
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return np.asarray(mask) > 0


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)) > 0


def overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.48) -> Image.Image:
    if mask.shape != (image.height, image.width):
        mask = resize_mask(mask, image.size)
    color_img = Image.new("RGB", image.size, color)
    masked = Image.composite(color_img, image, Image.fromarray(mask.astype(np.uint8) * 255))
    return Image.blend(image, masked, alpha)


def paste_fit(canvas: Image.Image, img: Image.Image, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    max_w, max_h = x1 - x0, y1 - y0
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    resized = img.resize(new_size, Image.Resampling.LANCZOS)
    canvas.paste(resized, (x0 + (max_w - resized.width) // 2, y0 + (max_h - resized.height) // 2))


def draw_center(draw: ImageDraw.ImageDraw, center_x: int, y: int, text: str, font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (bbox[2] - bbox[0]) / 2, y), text, fill="black", font=font)


def draw_vertical_label(canvas: Image.Image, text: str, center: tuple[int, int]) -> None:
    label = Image.new("RGBA", (230, 42), (255, 255, 255, 0))
    draw = ImageDraw.Draw(label)
    bbox = draw.textbbox((0, 0), text, font=SIDE_FONT)
    draw.text(((label.width - (bbox[2] - bbox[0])) / 2, 5), text, fill="black", font=SIDE_FONT)
    rotated = label.rotate(90, expand=True)
    canvas.paste(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2), rotated)


def clean_label(label: str) -> str:
    return label.split(",")[0].strip()


def crop_zoom(img: Image.Image, mask: np.ndarray, pad: int = 35) -> Image.Image:
    if mask.shape != (img.height, img.width):
        mask = resize_mask(mask, img.size)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(img.width, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.height, ys.max() + pad)
    return img.crop((x0, y0, x1, y1))


def render_case(name: str, spec: dict) -> Image.Image:
    headers, tiles, captions, gt_mask = case_tiles(spec)

    canvas = Image.new("RGB", (1507, 580), "white")
    draw = ImageDraw.Draw(canvas)
    left, col_w, gap = 82, 222, 17
    top_y, top_h = 70, 205
    zoom_y, zoom_h = 360, 135
    draw_vertical_label(canvas, "Full scene", (48, top_y + top_h // 2))
    draw_vertical_label(canvas, "Zoom target", (48, zoom_y + zoom_h // 2))

    for idx, header in enumerate(headers):
        x0 = left + idx * (col_w + gap)
        cx = x0 + col_w // 2
        draw_center(draw, cx, 10, header, HEADER_FONT)
        paste_fit(canvas, tiles[idx], (x0, top_y, x0 + col_w, top_y + top_h))
        draw_center(draw, cx, top_y + top_h + 14, captions[idx], LABEL_FONT)
        zoom = crop_zoom(tiles[idx], gt_mask)
        paste_fit(canvas, zoom, (x0, zoom_y, x0 + col_w, zoom_y + zoom_h))
        draw_center(draw, cx, zoom_y + zoom_h + 17, captions[idx], LABEL_FONT)

    canvas.save(OUT / f"{name}.pdf", "PDF", resolution=300)
    return canvas


def case_tiles(spec: dict) -> tuple[list[str], list[Image.Image], list[str], np.ndarray]:
    row = fresh_row(spec["filename"])
    labels = load_vocab(spec["vocab"])
    image = download_image(row["image"]["src"])
    target_obj = find_object(row, spec["raw_name"])
    gt_mask = object_mask(target_obj, row["image"]["width"], row["image"]["height"])
    if image.size != (row["image"]["width"], row["image"]["height"]):
        gt_mask = resize_mask(gt_mask, image.size)

    tiles = [image, overlay(image, gt_mask, (230, 48, 36), 0.45)]
    captions = ["", f"{spec['target']}[gt]"]
    pred_masks = [gt_mask, gt_mask]

    pred_gt_mask_size = None
    for _, method_key in METHODS:
        pred = Image.open(PRED_ROOT / method_key / spec["dataset"] / spec["filename"].replace(".jpg", ".png"))
        pred_arr = np.asarray(pred)
        gt_for_pred = resize_mask(gt_mask, pred.size)
        vals = pred_arr[gt_for_pred]
        majority = int(np.bincount(vals.ravel()).argmax())
        pred_mask = pred_arr == majority
        pred_mask_img = resize_mask(pred_mask, image.size)
        pred_label = clean_label(labels[majority - 1]) if 1 <= majority <= len(labels) else "background"
        tiles.append(overlay(image, pred_mask_img, (34, 190, 54), 0.48))
        mark = "OK" if norm(pred_label) == norm(spec["target"]) else "X"
        captions.append(f"{pred_label}[{mark}]")
        pred_masks.append(pred_mask_img)
        pred_gt_mask_size = gt_for_pred

    headers = ["Input", "GT"] + [name for name, _ in METHODS]
    return headers, tiles, captions, gt_mask


def render_overview_grid() -> None:
    cases = [
        ("E1 compact", "VOC-20", CASES["overview_window_ade150"]),
        ("E1 compact", "Ctx-59", CASES["overview_pot_ade150"]),
        ("E1 compact", "ADE-150", CASES["overview_people_ade150"]),
        ("E1 compact", "COCO", CASES["overview_computer_ade150"]),
        ("E2 large-vocab", "Ctx-459", CASES["case_study_large_vocab_shelf_ade847"]),
        ("E2 large-vocab", "ADE-847", CASES["case_study_large_vocab_jacket_ade847"]),
        ("Diagnostics", "Mismatch", CASES["case_study_mismatch_box_bookcase"]),
        ("Diagnostics", "Small obj.", CASES["case_study_small_object_bus_independent"]),
    ]
    prepared = []
    for group, title, spec in cases:
        headers, tiles, captions, _ = case_tiles(spec)
        prepared.append({"group": group, "title": title, "headers": headers, "tiles": tiles, "captions": captions})

    row_labels = prepared[0]["headers"]
    label_w = 88
    cell_w, cell_h = 238, 240
    cap_h = 42
    col_gap = 32
    row_gap = 18
    group_y = 42
    col_y = 92
    top = 142
    row_h = cell_h + cap_h + row_gap
    width = label_w + len(cases) * cell_w + (len(cases) - 1) * col_gap + 34
    height = top + len(row_labels) * row_h + 18
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(34)
    col_font = load_font(30)
    row_font = load_font(28)
    cap_font = load_font(22)

    def col_x(idx: int) -> int:
        return label_w + idx * (cell_w + col_gap)

    def span_center(start: int, end: int) -> int:
        x0 = col_x(start)
        x1 = col_x(end) + cell_w
        return (x0 + x1) // 2

    draw_center(draw, span_center(0, 3), group_y, "E1 compact", title_font)
    draw_center(draw, span_center(4, 5), group_y, "E2 large-vocab", title_font)
    draw_center(draw, span_center(6, 7), group_y, "Diagnostics", title_font)
    for idx, case in enumerate(prepared):
        draw_center(draw, col_x(idx) + cell_w // 2, col_y, case["title"], col_font)

    for row_idx, row_label in enumerate(row_labels):
        y = top + row_idx * row_h
        draw_vertical_label(canvas, row_label, (35, y + cell_h // 2))
        for col_idx, case in enumerate(prepared):
            tile_idx = case["headers"].index(row_label)
            x = col_x(col_idx)
            paste_fit(canvas, case["tiles"][tile_idx], (x, y, x + cell_w, y + cell_h))
            caption = case["captions"][tile_idx]
            if caption:
                draw_center(draw, x + cell_w // 2, y + cell_h + 6, caption, cap_font)

    for filename in [
        "qualitative_comparison_new_cases.pdf",
        "qualitative_comparison.pdf",
        "case_study_grid_overview.pdf",
    ]:
        canvas.save(OUT / filename, "PDF", resolution=300)


def render_combined() -> None:
    top = render_case("case_study_large_vocab_shelf_ade847", CASES["case_study_large_vocab_shelf_ade847"])
    bottom = render_case("case_study_large_vocab_jacket_ade847", CASES["case_study_large_vocab_jacket_ade847"])
    pad, label_h = 28, 42
    combined = Image.new("RGB", (top.width, top.height + bottom.height + 2 * label_h + pad), "white")
    draw = ImageDraw.Draw(combined)
    draw.text((40, 8), "(a) ADE-847 / shelf: large-vocabulary target renamed as scene context", fill=(0, 0, 0), font=PANEL_FONT)
    combined.paste(top, (0, label_h))
    y = label_h + top.height + pad
    draw.text((40, y), "(b) ADE-847 / jacket: fine-grained target absorbed by nearby surfaces", fill=(0, 0, 0), font=PANEL_FONT)
    combined.paste(bottom, (0, y + label_h))
    combined.save(OUT / "case_study_large_vocab_combined.pdf", "PDF", resolution=300)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    render_combined()
    render_case("case_study_mismatch_box_bookcase", CASES["case_study_mismatch_box_bookcase"])
    render_case("case_study_small_object_bus_independent", CASES["case_study_small_object_bus_independent"])
    render_overview_grid()


if __name__ == "__main__":
    main()
