#!/usr/bin/env bash
# Run CLIPtrase on TF-OVOS E1/E2 datasets and capture mIoU results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
REPO="$ROOT/third_party/official_methods/CLIPtrase"
LOGS="$ROOT/runs/logs"

mkdir -p "$LOGS"

log() { echo "[cliptrase] $* $(date -Is)"; }

wait_for_gpu_idle() {
  # CLIPtrase needs ~3GB; wait only if less than 8GB free
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt 8000 ]]; do
    log "waiting for GPU memory (need 8GB free)..."
    sleep 60
  done
}

install_deps() {
  "$PY" -c "import einops" 2>/dev/null || "$PY" -m pip install einops -q
  "$PY" -c "import sklearn" 2>/dev/null || "$PY" -m pip install scikit-learn -q
  "$PY" -c "import cv2" 2>/dev/null || "$PY" -m pip install opencv-python-headless -q
}

run_cliptrase() {
  local dataset="$1"
  local log="$LOGS/official_cliptrase_${dataset}.log"
  local legacy_log="$LOGS/official_cliptrase_${dataset}_e1.log"

  if { [[ -f "$log" ]] && grep -q "results:" "$log" 2>/dev/null; } ||      { [[ -f "$legacy_log" ]] && grep -q "results:" "$legacy_log" 2>/dev/null; }; then
    log "skip $dataset; already complete"
    return
  fi

  wait_for_gpu_idle
  log "start CLIPtrase $dataset"
  (
    cd "$REPO"
    "$PY" -c "
import sys, torch
sys.path.insert(0, '.')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import clip_utils
from clip_self_correlation import self_clip

clip_type = 'ViT-B/16'
clip_model, _ = clip_utils.load(clip_type, image_size=224)
clip_model = clip_model.to(device)
print('load clip success!')

with torch.no_grad():
    self_clip(clip_model, '$dataset', image_size=224, eps=0.7, min=3)
    self_clip(clip_model, '$dataset', image_size=336, eps=1.1, min=7)
"
  ) 2>&1 | tee "$log" || {
    log "FAILED CLIPtrase $dataset; see $log"
    return 0
  }
  log "done CLIPtrase $dataset"
}

log "installing dependencies..."
install_deps

for dataset in VOC20 PC59 ADE150 COCO171_val PC459 ADEfull; do
  run_cliptrase "$dataset"
done

"$PY" "$ROOT/scripts/analyze_official_results.py" 2>/dev/null || true

log "CLIPtrase queue finished"
