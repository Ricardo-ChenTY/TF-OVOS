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
OUT="$ROOT/runs/trained_refs/odise/ctx59_fixed"
LOG="$ROOT/runs/logs/odise_ctx59_fixed.log"
mkdir -p "$OUT" "$ROOT/runs/logs"

wait_for_gpu_idle() {
  local need_mb="${1:-50000}"
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt "$need_mb" ]]; do
    echo "[odise-ctx59-fixed] waiting for GPU memory (need ${need_mb}MB free) $(date -Is)"
    sleep 60
  done
}

wait_for_gpu_idle 60000

echo "[odise-ctx59-fixed] start $(date -Is)" | tee "$LOG"
$PY "$WRAP" tools/train_net.py   --config-file configs/Panoptic/odise_label_open_ctx59_fixed.py   --num-gpus 1 --eval-only   --init-from "$CKPT"   --output "$OUT" 2>&1 | tee -a "$LOG"
echo "[odise-ctx59-fixed] done $(date -Is)" | tee -a "$LOG"
