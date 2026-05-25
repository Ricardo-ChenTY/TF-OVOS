#!/usr/bin/env bash
# Debug-only MaskCLIP-style attention adapter on E1 + E2 datasets.
#
# This is not the official MaskCLIP repo/evaluation path and must not be used
# for official reproduction rows. Set ALLOW_DEBUG_MASKCLIP_ADAPTER=1 only when
# intentionally collecting exploratory adapter numbers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"

mkdir -p "$LOGS"

log() { echo "[maskclip-debug-adapter] $* $(date -Is)"; }

if [[ "${ALLOW_DEBUG_MASKCLIP_ADAPTER:-0}" != "1" ]]; then
  log "refusing to run: maskclip_attn is a TF-OVOS debug adapter, not official MaskCLIP"
  log "use the official MaskCLIP repo/env for official rows; set ALLOW_DEBUG_MASKCLIP_ADAPTER=1 only for diagnostics"
  exit 1
fi

metrics_done() {
  local dataset="$1"
  local metrics="$ROOT/runs/maskclip_attn/${dataset}_val/metrics.json"
  [[ -f "$metrics" ]] && grep -qF '"miou"' "$metrics"
}

wait_for_gpu_idle() {
  local need_mb="${1:-8000}"
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt "$need_mb" ]]; do
    log "waiting for GPU memory (need ${need_mb}MB free)..."
    sleep 60
  done
}

run_adapter() {
  local dataset="$1"
  local log="$LOGS/debug_maskclip_attn_${dataset}_e1.log"

  if metrics_done "$dataset"; then
    log "skip $dataset; already complete"
    return
  fi

  wait_for_gpu_idle 8000
  log "start maskclip_attn $dataset"
  "$PY" -m tf_ovos.run_benchmark \
    --method maskclip_attn \
    --dataset "${dataset}_val" \
    --num-shards 1 \
    --skip-existing \
    2>&1 | tee "$log" || {
    log "FAILED maskclip_attn $dataset; see $log"
    return 0
  }
  log "done maskclip_attn $dataset"
}

run_adapter_e2() {
  local dataset="$1"
  local suffix="$2"
  local log="$LOGS/debug_maskclip_attn_${suffix}_e2.log"

  if metrics_done "$dataset"; then
    log "skip $dataset E2; already complete"
    return
  fi

  if [[ "$dataset" == "ade20k847" ]]; then
    wait_for_gpu_idle 70000
  else
    wait_for_gpu_idle 35000
  fi
  log "start maskclip_attn $dataset E2"
  "$PY" -m tf_ovos.run_benchmark \
    --method maskclip_attn \
    --dataset "${dataset}_val" \
    --num-shards 1 \
    --skip-existing \
    2>&1 | tee "$log" || {
    log "FAILED maskclip_attn $dataset E2; see $log"
    return 0
  }
  log "done maskclip_attn $dataset E2"
}

# E1: standard 4 datasets
run_adapter voc20
run_adapter context59
run_adapter ade20k150
run_adapter coco_stuff171

# E2: large-vocab
run_adapter_e2 context459 context459
run_adapter_e2 ade20k847 ade847

"$PY" "$ROOT/scripts/analyze_official_results.py" 2>/dev/null || true

log "debug MaskCLIP adapter queue finished"
