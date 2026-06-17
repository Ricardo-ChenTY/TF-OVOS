from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tf_ovcos_full_results_20260526/runs/analysis/case_study_mismatch_signboard.png"
OUT = ROOT / "TMLR26_Bench/figures/protocol_real_examples"

LEFT = 82
COL_W = 222
GAP = 17
TOP_Y = 70
TOP_H = 205
ZOOM_Y = 360
ZOOM_H = 135


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT = load_font(28)
FONT_SMALL = load_font(22)
FONT_XS = load_font(16)
FONT_BOLD = load_font(28, bold=True)


def cell_box(col: int, row: str = "full") -> tuple[int, int, int, int]:
    x0 = LEFT + col * (COL_W + GAP)
    if row == "zoom":
        return (x0, ZOOM_Y, x0 + COL_W, ZOOM_Y + ZOOM_H)
    return (x0, TOP_Y, x0 + COL_W, TOP_Y + TOP_H)


def fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    scale = min(size[0] / img.width, size[1] / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def crop_cell(src: Image.Image, col: int, row: str = "full", size: tuple[int, int] = (260, 160)) -> Image.Image:
    return fit(src.crop(cell_box(col, row)), size)


def add_label(img: Image.Image, label: str) -> Image.Image:
    pad = 10
    label_h = 36
    out = Image.new("RGB", (img.width, img.height + label_h), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
    draw.text(((out.width - (bbox[2] - bbox[0])) // 2, img.height + pad // 2), label, fill=(20, 20, 20), font=FONT_SMALL)
    return out


def bordered(img: Image.Image, color: tuple[int, int, int] = (90, 90, 90)) -> Image.Image:
    out = Image.new("RGB", (img.width + 4, img.height + 4), color)
    out.paste(img, (2, 2))
    return out


def make_stack(src: Image.Image) -> Image.Image:
    base = crop_cell(src, 0, "full", (230, 150))
    out = Image.new("RGBA", (280, 190), (255, 255, 255, 0))
    for dx, dy in [(0, 0), (15, 12), (30, 24)]:
        tile = bordered(base).convert("RGBA")
        out.paste(tile, (dx, dy), tile)
    return out.convert("RGB")


def make_mask_stack(src: Image.Image) -> Image.Image:
    base = crop_cell(src, 4, "zoom", (105, 85))
    middle = crop_cell(src, 3, "zoom", (105, 85))
    back = crop_cell(src, 2, "zoom", (105, 85))
    out = Image.new("RGBA", (145, 115), (255, 255, 255, 0))
    for tile, pos in [(back, (0, 0)), (middle, (12, 10)), (base, (24, 20))]:
        tile_rgba = bordered(tile).convert("RGBA")
        out.paste(tile_rgba, pos, tile_rgba)
    return out.convert("RGB")


def make_patch_grid(src: Image.Image, cols: list[int]) -> Image.Image:
    cell = 48
    gap = 4
    out = Image.new("RGB", (2 * cell + gap, 2 * cell + gap), "white")
    crops = [fit(src.crop(cell_box(col, "zoom")), (cell, cell)) for col in cols]
    for idx, img in enumerate(crops):
        x = (idx % 2) * (cell + gap)
        y = (idx // 2) * (cell + gap)
        out.paste(img, (x, y))
    return bordered(out)


def make_vocab_card() -> Image.Image:
    out = Image.new("RGB", (240, 170), (255, 250, 228))
    draw = ImageDraw.Draw(out)
    draw.rounded_rectangle((3, 3, 236, 166), radius=10, outline=(155, 132, 78), width=3, fill=(255, 250, 228))
    draw.text((18, 15), "ADE-150 labels", fill=(40, 40, 40), font=FONT_BOLD)
    labels = ["signboard", "bar", "barrel", "wall", "person"]
    for i, text in enumerate(labels):
        draw.text((28, 52 + i * 22), f"- {text}", fill=(30, 30, 30), font=FONT_SMALL)
    return out


def make_run_card() -> Image.Image:
    out = Image.new("RGB", (320, 205), "white")
    draw = ImageDraw.Draw(out)
    draw.rectangle((3, 3, 316, 201), outline=(135, 135, 135), width=3, fill=(250, 250, 250))
    draw.text((16, 12), "Run card", fill=(30, 30, 30), font=FONT_BOLD)
    rows = [
        ("dataset", "ADE-150 val"),
        ("image", "ADE_val_00000068"),
        ("prompt", "a photo of {c}"),
        ("resize", "fixed"),
        ("threshold", "fixed"),
    ]
    y = 52
    for key, value in rows:
        draw.line((14, y - 6, 306, y - 6), fill=(210, 210, 210), width=1)
        draw.text((18, y), key, fill=(45, 45, 45), font=FONT_XS)
        draw.text((118, y), value, fill=(45, 45, 45), font=FONT_XS)
        y += 27
    return out


def make_pair(src: Image.Image, left_col: int, right_col: int, label: str) -> Image.Image:
    a = crop_cell(src, left_col, "zoom", (150, 112))
    b = crop_cell(src, right_col, "zoom", (150, 112))
    out = Image.new("RGB", (326, 150), "white")
    out.paste(bordered(a), (0, 0))
    out.paste(bordered(b), (172, 0))
    draw = ImageDraw.Draw(out)
    draw.text((154, 42), "+", fill=(35, 35, 35), font=FONT_BOLD)
    bbox = draw.textbbox((0, 0), label, font=FONT_SMALL)
    draw.text(((out.width - (bbox[2] - bbox[0])) // 2, 122), label, fill=(30, 30, 30), font=FONT_SMALL)
    return out


def make_three_panel(src: Image.Image, cols: list[int], labels: list[str], filename: str) -> None:
    tiles = [add_label(crop_cell(src, col, "zoom", (165, 120)), label) for col, label in zip(cols, labels)]
    out = Image.new("RGB", (len(tiles) * 185 - 20, 166), "white")
    for i, tile in enumerate(tiles):
        out.paste(bordered(tile), (i * 185, 0))
    out.save(OUT / filename)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = Image.open(SRC).convert("RGB")

    assets = {
        "target_images_real_stack.png": make_stack(src),
        "target_images_icon_only.png": make_stack(src),
        "dense_vlm_icon_only.png": make_patch_grid(src, [0, 5, 3, 2]),
        "vfm_assisted_icon_only.png": make_patch_grid(src, [0, 2, 3, 4]),
        "proposal_stack_icon_only.png": make_mask_stack(src),
        "candidate_vocabulary_real_card.png": make_vocab_card(),
        "run_card_real_table.png": make_run_card(),
        "vlm_vfm_example_pair_real.png": make_pair(src, 0, 2, "real case: signboard -> bar[X]"),
    }
    for filename, img in assets.items():
        img.save(OUT / filename)

    make_three_panel(src, [0, 5, 3], ["image", "SCLIP map", "NACLIP map"], "dense_vlm_maps_real.png")
    make_three_panel(src, [0, 2, 3], ["image", "CorrCLIP", "NACLIP"], "vfm_assisted_dense_maps_real.png")
    make_three_panel(src, [0, 4, 4], ["image", "SAM mask", "named mask"], "proposal_vlm_naming_real.png")

    make_contact_sheet()


def make_contact_sheet() -> None:
    items = [
        ("Target images", "target_images_real_stack.png"),
        ("Candidate vocabulary", "candidate_vocabulary_real_card.png"),
        ("Run card", "run_card_real_table.png"),
        ("Dense VLM maps", "dense_vlm_maps_real.png"),
        ("VFM-assisted dense maps", "vfm_assisted_dense_maps_real.png"),
        ("Proposal + VLM naming", "proposal_vlm_naming_real.png"),
    ]
    width, margin, gap = 720, 28, 22
    title_h = 40
    loaded = [(title, Image.open(OUT / filename).convert("RGB")) for title, filename in items]
    height = margin + sum(title_h + img.height + gap for _, img in loaded) - gap + margin
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = margin
    for title, img in loaded:
        draw.text((margin, y), title, fill=(20, 20, 20), font=FONT_BOLD)
        y += title_h
        sheet.paste(img, (margin, y))
        y += img.height + gap
    sheet.save(OUT / "protocol_real_examples_contact_sheet.png")


if __name__ == "__main__":
    main()
