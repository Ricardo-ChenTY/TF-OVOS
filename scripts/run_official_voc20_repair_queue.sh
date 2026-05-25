#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
MARKERS="$ROOT/runs/official/_repair_markers"

mkdir -p "$LOGS" "$MARKERS"

log() { echo "[official-voc20-repair] $* $(date -Is)"; }

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

run_naclip() {
  local marker="$MARKERS/naclip_voc20_officialnames.done"
  local log_path="$LOGS/official_naclip_voc20_officialnames_fixed.log"
  local work_dir="$ROOT/runs/official/naclip/voc20_officialnames_fixed"
  if [[ -f "$marker" ]]; then
    log "skip NACLIP VOC20 officialnames repair; marker exists"
    return
  fi

  wait_for_gpu_idle
  log "start NACLIP VOC20 officialnames repair"
  (
    cd "$ROOT/third_party/official_methods/NACLIP"
    "$PY" eval.py \
      --config configs/cfg_tfovos_voc20_officialnames.py \
      --work-dir "$work_dir"
  ) > "$log_path" 2>&1 && {
    touch "$marker"
    log "done NACLIP VOC20 officialnames repair"
  } || {
    log "FAILED NACLIP VOC20 officialnames repair; see $log_path"
  }
}

run_corrclip() {
  local marker="$MARKERS/corrclip_voc20_officialnames.done"
  local log_path="$LOGS/official_corrclip_voc20_officialnames_fixed.log"
  if [[ -f "$marker" ]]; then
    log "skip CorrCLIP VOC20 officialnames repair; marker exists"
    return
  fi

  wait_for_gpu_idle
  log "start CorrCLIP VOC20 officialnames repair"
  (
    cd "$ROOT/third_party/official_methods/CorrCLIP"
    "$PY" eval.py \
      --config configs/cfg_tfovos_voc20_officialnames.py
  ) > "$log_path" 2>&1 && {
    touch "$marker"
    log "done CorrCLIP VOC20 officialnames repair"
  } || {
    log "FAILED CorrCLIP VOC20 officialnames repair; see $log_path"
  }
}

"$PY" "$ROOT/scripts/prepare_official_mmseg_configs.py"
"$PY" "$ROOT/scripts/prepare_official_voc20_officialnames_configs.py"

# Let the proposal official-name VOC20 reruns finish first; then repair the
# official rows before MaskCLIP and the main proposal queue continue.
wait_for_tmux_done proposal_voc20_officialnames_queue

run_naclip
run_corrclip

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
log "official VOC20 repair queue finished"
