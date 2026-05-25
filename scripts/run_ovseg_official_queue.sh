#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/tianyi/TF-OVCOS
while tmux has-session -t san_official_all 2>/dev/null; do
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] waiting for san_official_all..."
  sleep 60
done
cd "$ROOT/third_party/official_methods/ov-seg"
export DETECTRON2_DATASETS="$ROOT/data/detectron2_refs"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PY=/data/tianyi/conda_envs/ovseg-official/bin/python
WRAP="$ROOT/scripts/run_with_pillow_compat.py"
CKPT="$ROOT/checkpoints/ovseg/ovseg_swinbase_vitL14_ft_mpt.pth"
OUT="$ROOT/runs/trained_refs/ovseg"
LOG="$ROOT/runs/logs"
mkdir -p "$OUT/ade" "$OUT/context" "$OUT/voc" "$LOG"

$PY "$WRAP" train_net.py --num-gpu 1 --eval-only   --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml   MODEL.WEIGHTS "$CKPT"   DATASETS.TEST '("ade20k_sem_seg_val","ade20k_full_sem_seg_val")'   OUTPUT_DIR "$OUT/ade" 2>&1 | tee "$LOG/ovseg_official_ade.log"

$PY "$WRAP" train_net.py --num-gpu 1 --eval-only   --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml   MODEL.WEIGHTS "$CKPT"   MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.6   DATASETS.TEST '("pascal_context_59_sem_seg_val","pascal_context_459_sem_seg_val")'   OUTPUT_DIR "$OUT/context" 2>&1 | tee "$LOG/ovseg_official_context.log"

$PY "$WRAP" train_net.py --num-gpu 1 --eval-only   --config-file configs/ovseg_swinB_vitL_bs32_120k.yaml   MODEL.WEIGHTS "$CKPT"   MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.45   DATASETS.TEST '("pascalvoc20_sem_seg_val",)'   OUTPUT_DIR "$OUT/voc" 2>&1 | tee "$LOG/ovseg_official_voc.log"
