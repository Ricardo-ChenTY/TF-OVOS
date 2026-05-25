#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
VOCAB="$ROOT/configs/vocab/voc20_officialnames.txt"
RUN_ROOT="$ROOT/runs/proposal_officialnames"

mkdir -p "$LOGS" "$RUN_ROOT"

log() { echo "[proposal-voc20-officialnames] $* $(date -Is)"; }

wait_for_tmux_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    log "waiting for current GPU jobs"
    sleep 300
  done
}

metrics_done() {
  local method="$1"
  local metrics="$RUN_ROOT/${method}/voc20_val/metrics.json"
  [[ -f "$metrics" ]] && grep -qF '"miou"' "$metrics"
}

run_method() {
  local method="$1"
  local log_path="$LOGS/proposal_${method}_voc20_officialnames.log"
  if metrics_done "$method"; then
    log "skip $method VOC20 officialnames; already complete"
    return
  fi
  wait_for_gpu_idle
  log "start $method VOC20 officialnames"
  (
    cd "$ROOT"
    "$PY" -m tf_ovos.run_benchmark \
      --method "$method" \
      --dataset voc20_val \
      --vocab "$VOCAB" \
      --run-root "$RUN_ROOT" \
      --num-shards 1
  ) > "$log_path" 2>&1 || {
    log "FAILED $method VOC20 officialnames; see $log_path"
    return 0
  }
  log "done $method VOC20 officialnames"
}

wait_for_tmux_done official_voc20_officialnames_queue
run_method sam_amg_clip
run_method sam_amg_siglip
"$PY" -m tf_ovos.summarize_results --run-root "$RUN_ROOT" --out-dir "$RUN_ROOT/tables" || true
log "proposal VOC20 officialnames queue finished"
