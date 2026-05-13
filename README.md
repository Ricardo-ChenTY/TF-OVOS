# TF-OVCOS Benchmark Skeleton

This repository is a lightweight harness for the TF-OVCOS proposal:
training-free open-vocabulary camouflaged object segmentation.

The benchmark contract is intentionally simple. Every method adapter receives
an image and a fixed vocabulary, then writes exactly one prediction per image:

```json
{"image_id": "xxx", "mask_path": "pred_masks/xxx.png", "label": "frog", "score": 0.73}
```

The evaluator consumes the same prediction format for E1, E2, and E3.

## Server Quickstart

On a fresh Linux server:

```bash
git clone https://github.com/Ricardo-ChenTY/TF-OVCOS.git
cd TF-OVCOS
bash scripts/setup_conda.sh
```

Manual equivalent:

```bash
conda env create -f environment.yml
conda activate tf-ovcos
python -m pytest tests
python scripts/smoke_test.py
python -m tf_ovcos.check_ready
```

This verifies the core benchmark harness with toy data and the debug adapter.
Real GPU methods still need their own third-party repositories, CUDA-matched
PyTorch wheels, model weights, and adapter implementations.

`python -m tf_ovcos.check_ready` reports which vocab, manifest, and adapter
items are complete. Missing real data and planned adapters are expected before
the GPU machine is provisioned.

Create the ignored local workspace directories with:

```bash
bash scripts/prepare_workspace.sh
```

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

Current adapter status can be checked with:

```bash
python -m tf_ovcos.run_method --list-methods
```

## Reproducible Pipeline

Run one method on one or more configured datasets:

```bash
python -m tf_ovcos.run_benchmark \
  --method debug_copy_gt \
  --dataset ovcamo_te \
  --limit 20 \
  --num-shards 8 \
  --skip-existing
```

For external E3 targets, omit `--dataset` to run every dataset in
`configs/benchmark.yaml`. Drop `--limit` for full runs. Class-aware datasets use
`ovcamo_61_unseen`; mask-only external datasets use `ovcamo_75`.

Summarize E1/E2/E3/E4-style outputs after runs finish:

```bash
python -m tf_ovcos.summarize_results --method debug_copy_gt --out-dir runs/tables
```

Each shard writes `runtime.json` with wall-clock seconds and seconds/image. Real
adapters can extend that file with peak memory, model calls, and module-specific
metadata.

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
