import yaml

from tf_ovcos.check_ready import check_ready


def test_check_ready_reports_placeholder_vocab(tmp_path):
    root = tmp_path
    vocab_dir = root / "configs" / "vocab"
    vocab_dir.mkdir(parents=True)
    (vocab_dir / "ovcamo_75.txt").write_text("# Fill with labels\n", encoding="utf-8")

    cfg = {
        "vocabularies": {"ovcamo_75": "configs/vocab/ovcamo_75.txt"},
        "datasets": {},
    }
    config_path = root / "configs" / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (root / "configs" / "methods").mkdir(parents=True)

    rows = check_ready(root, config_path)
    assert any(row.level == "FAIL" and row.item == "vocab:ovcamo_75" for row in rows)


def test_check_ready_accepts_complete_toy_manifest(tmp_path):
    root = tmp_path
    vocab_dir = root / "configs" / "vocab"
    data_dir = root / "data" / "manifests"
    image_dir = root / "data" / "raw" / "toy" / "images"
    mask_dir = root / "data" / "raw" / "toy" / "masks"
    method_dir = root / "configs" / "methods"
    for path in (vocab_dir, data_dir, image_dir, mask_dir, method_dir):
        path.mkdir(parents=True)

    (vocab_dir / "toy.txt").write_text("frog\n", encoding="utf-8")
    (image_dir / "a.jpg").write_bytes(b"placeholder")
    (mask_dir / "a.png").write_bytes(b"placeholder")
    manifest = data_dir / "toy.jsonl"
    manifest.write_text(
        '{"image_id": "a", "image_path": "../raw/toy/images/a.jpg", "mask_path": "../raw/toy/masks/a.png", "label": "frog"}\n',
        encoding="utf-8",
    )
    cfg = {
        "vocabularies": {"toy": "configs/vocab/toy.txt"},
        "datasets": {"toy": {"manifest": "data/manifests/toy.jsonl", "task": "class-aware"}},
    }
    config_path = root / "configs" / "benchmark.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    rows = check_ready(root, config_path)
    assert any(row.level == "OK" and row.item == "manifest:toy" for row in rows)
