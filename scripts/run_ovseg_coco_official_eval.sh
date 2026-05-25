#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/tianyi/TF-OVCOS
cd "$ROOT/third_party/official_methods/ov-seg"
export DETECTRON2_DATASETS="$ROOT/data/detectron2_refs"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/data/tianyi/conda_envs/ovseg-official/bin/python
WRAP="$ROOT/scripts/run_with_pillow_compat.py"
CKPT="$ROOT/checkpoints/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth"
OUT="$ROOT/runs/trained_refs/ovseg/coco"
LOG="$ROOT/runs/logs"
mkdir -p "$OUT" "$LOG"
$PY "$WRAP" train_net.py --num-gpu 1 --eval-only \
  --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml \
  MODEL.WEIGHTS "$CKPT" \
  DATASETS.TEST '("coco_2017_val_stuff_sem_seg",)' \
  OUTPUT_DIR "$OUT" 2>&1 | tee "$LOG/ovseg_official_coco.log"
