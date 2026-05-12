# TF-OVCOS Benchmark Skeleton

This repository is a lightweight harness for the TF-OVCOS proposal:
training-free open-vocabulary camouflaged object segmentation.

The benchmark contract is intentionally simple. Every method adapter receives
an image and a fixed vocabulary, then writes exactly one prediction per image:

```json
{"image_id": "xxx", "mask_path": "pred_masks/xxx.png", "label": "frog", "score": 0.73}
```

The evaluator consumes the same prediction format for E1, E2, and E3.

## Suggested Build Order

1. **Core harness first**
   - dataset manifests under `data/manifests/`
   - method outputs under `runs/<method>/<split>/predictions.jsonl`
   - metrics via `python -m tf_ovcos.eval`

2. **Minimal baseline**
   - add one simple adapter first, e.g. `groundingdino_sam` or `sam_amg_clip`
   - verify output masks and labels on 20 images
   - run E1/E2 metrics end to end

3. **Scale method rows**
   - dense-map methods: MaskCLIP, SCLIP, NACLIP, ProxyCLIP
   - detector + SAM: GroundingDINO + SAM/SAM2
   - proposal + naming: SAM-AMG + CLIP/SigLIP, DINOv2 + SAM + VLM
   - diffusion/reference rows last because setup/runtime is heavier

Before requesting a large GPU machine, check the Chinese preparation checklist
in `docs/gpu_request_checklist_zh.md`. It lists the dataset, vocabulary,
manifest, storage, and first-baseline items that should be ready before moving
to full GPU inference.

## Dataset Manifest

Create one JSONL manifest per split:

```json
{"image_id": "0001", "image_path": "data/OVCamo-TE/images/0001.jpg", "mask_path": "data/OVCamo-TE/masks/0001.png", "label": "frog"}
```

For external E3 mask-only datasets, `label` can be omitted or set to `null`.

## Commands

```powershell
python -m pip install -e .
python -m tf_ovcos.make_manifest --image-dir data/raw/OVCamo/TE/images --mask-dir data/raw/OVCamo/TE/masks --label-source data/raw/OVCamo/sample_info.json --out data/manifests/ovcamo_te.jsonl --require-labels
python -m tf_ovcos.make_shards --manifest data/manifests/ovcamo_te.jsonl --num-shards 8 --out-dir data/manifests/shards/ovcamo_te
python -m tf_ovcos.run_method --method debug_copy_gt --manifest data/manifests/shards/ovcamo_te/part-000.jsonl --out runs/debug_copy_gt/ovcamo_te/shards/part-000
python -m tf_ovcos.merge_predictions --manifest data/manifests/ovcamo_te.jsonl --shards-dir runs/debug_copy_gt/ovcamo_te/shards --out runs/debug_copy_gt/ovcamo_te/predictions.jsonl
python -m tf_ovcos.eval --manifest data/manifests/ovcamo_te.jsonl --predictions runs/sam_amg_clip/ovcamo_te/predictions.jsonl --task class-aware --out runs/sam_amg_clip/ovcamo_te/metrics.json
python -m tf_ovcos.eval --manifest data/manifests/nc4k.jsonl --predictions runs/sam_amg_clip/nc4k/predictions.jsonl --task mask-only --out runs/sam_amg_clip/nc4k/metrics.json
```

`debug_copy_gt` and `debug_empty` are pipeline smoke-test adapters, not paper
methods. Real methods should be added under `src/tf_ovcos/adapters/` and
registered in `tf_ovcos.run_method`.

## Public Code Starting Points

- MaskCLIP: <https://github.com/chongzhou96/MaskCLIP>
- SCLIP: <https://github.com/wangf3014/SCLIP>
- NACLIP: <https://github.com/sinahmr/NACLIP>
- ProxyCLIP: <https://github.com/mc-lan/ProxyCLIP>
- Grounded-Segment-Anything: <https://github.com/IDEA-Research/Grounded-Segment-Anything>
- FreeDA: <https://github.com/aimagelab/freeda>

Keep these method repositories as external submodules or sibling folders. The
benchmark code should only depend on their exported masks/labels, not their
internal evaluation scripts.
