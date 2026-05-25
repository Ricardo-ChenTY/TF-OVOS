#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/tianyi/TF-OVCOS
cd "$ROOT/third_party/official_methods/ODISE"
export DETECTRON2_DATASETS="$ROOT/data/detectron2_refs"
export PYTHONPATH="$PWD:$PWD/third_party/Mask2Former:${PYTHONPATH:-}"
export HF_HOME=/data/tianyi/home_moved/.cache/huggingface
export TRANSFORMERS_CACHE=/data/tianyi/home_moved/.cache/huggingface/hub
PY=/data/tianyi/conda_envs/odise-official/bin/python
WRAP="$ROOT/scripts/run_with_pillow_compat.py"
CKPT="$ROOT/checkpoints/odise/odise_label_coco_50e-b67d2efc.pth"
OUT="$ROOT/runs/trained_refs/odise/partial"
LOG="$ROOT/runs/logs"
mkdir -p "$OUT" "$LOG"
$PY "$WRAP" tools/train_net.py \
  --config-file configs/Panoptic/odise_label_open_partial.py \
  --num-gpus 1 --eval-only \
  --init-from "$CKPT" \
  --output "$OUT" 2>&1 | tee "$LOG/odise_official_partial.log"
