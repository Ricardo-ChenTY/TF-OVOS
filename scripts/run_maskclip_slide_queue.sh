#!/usr/bin/env bash
# Debug-only sliding-window MaskCLIP-style attention adapter.
#
# This is not the official MaskCLIP repo/evaluation path and must not be used
# for official reproduction rows. Set ALLOW_DEBUG_MASKCLIP_ADAPTER=1 only when
# intentionally collecting exploratory adapter numbers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"

mkdir -p "$LOGS"

log() { echo "[maskclip-slide-debug] $* $(date -Is)"; }

if [[ "${ALLOW_DEBUG_MASKCLIP_ADAPTER:-0}" != "1" ]]; then
  log "refusing to run: maskclip_attn_slide is a TF-OVOS debug adapter, not official MaskCLIP"
  log "use the official MaskCLIP repo/env for official rows; set ALLOW_DEBUG_MASKCLIP_ADAPTER=1 only for diagnostics"
  exit 1
fi

metrics_done() {
  local dataset="$1"
  local metrics="$ROOT/runs/maskclip_attn_slide/${dataset}_val/metrics.json"
  [[ -f "$metrics" ]] && grep -qF '"miou"' "$metrics"
}

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  local need_mb="${1:-35000}"
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt "$need_mb" ]]; do
    log "waiting for GPU memory (need ${need_mb}MB free)"
    sleep 60
  done
}

run_dataset() {
  local dataset="$1"
  local suffix="$2"
  local phase="$3"
  local log_path="$LOGS/maskclip_slide_${suffix}_${phase}.log"

  if metrics_done "$dataset"; then
    log "skip $dataset; already complete"
    return
  fi

  wait_for_gpu_idle 35000
  log "start maskclip_attn_slide $dataset $phase"
  "$PY" -m tf_ovos.run_benchmark \
    --method maskclip_attn_slide \
    --dataset "${dataset}_val" \
    --num-shards 1 \
    --skip-existing \
    2>&1 | tee "$log_path" || {
      log "FAILED maskclip_attn_slide $dataset; see $log_path"
      return 0
    }
  log "done maskclip_attn_slide $dataset $phase"
}

wait_for_session official_e2_retry

run_dataset voc20 voc20 e1
run_dataset context59 context59 e1
run_dataset ade20k150 ade20k150 e1
run_dataset coco_stuff171 coco_stuff171 e1
run_dataset context459 context459 e2
run_dataset ade20k847 ade847 e2

"$PY" "$ROOT/scripts/analyze_official_results.py" 2>/dev/null || true

log "maskclip slide queue finished"
