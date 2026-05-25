#!/usr/bin/env bash
# Run SC-CLIP under the shared TF-OVOS E1/E2 protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
RUNS="$ROOT/runs/official"
REPO="$ROOT/third_party/official_methods/SC-CLIP"

mkdir -p "$LOGS" "$RUNS"

log() { echo "[scclip] $* $(date -Is)"; }

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session: $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  local need_mb="${1:-45000}"
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt "$need_mb" ]]; do
    log "waiting for GPU memory (need ${need_mb}MB free)..."
    sleep 120
  done
}

done_log() {
  local log_file="$1"
  local total="$2"
  [[ -f "$log_file" ]] || return 1
  if grep -Eq "Traceback|OutOfMemory|CUDA out of memory|FAILED|failed" "$log_file"; then
    return 1
  fi
  grep -Eq "Iter\\(test\\).*\\[ *${total}/${total}\\]" "$log_file" && \
    grep -Eq "aAcc: +[0-9.]+ +mIoU: +[0-9.]+ +mAcc:" "$log_file"
}

run_dataset() {
  local dataset="$1"
  local total="$2"
  local config="configs/cfg_tfovos_${dataset}.py"
  local work_dir="$RUNS/scclip/${dataset}"
  local log_file="$LOGS/official_scclip_${dataset}.log"

  if done_log "$log_file" "$total"; then
    log "skip $dataset; already complete"
    return
  fi

  wait_for_gpu_idle 45000
  log "start $dataset"
  (
    cd "$REPO"
    "$PY" eval.py --config "$config" --work-dir "$work_dir"
  ) > "$log_file" 2>&1 || {
    log "FAILED $dataset; see $log_file"
    return 0
  }
  log "done $dataset"
}

"$PY" "$ROOT/scripts/prepare_scclip_configs.py"

wait_for_session odise_partial_fixed
wait_for_session ovseg_context59_fix_after_odise
wait_for_session official_dense_e2_after_refs
wait_for_session dinov2_sam_after_e2
wait_for_session trident_cass_after_dinov2

for dataset in voc20 context59 ade20k coco_stuff164k context459 ade847; do
  case "$dataset" in
    voc20) total=1449 ;;
    context59) total=5105 ;;
    ade20k) total=2000 ;;
    coco_stuff164k) total=5000 ;;
    context459) total=5105 ;;
    ade847) total=2000 ;;
  esac
  run_dataset "$dataset" "$total"
done

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
log "SC-CLIP queue finished"
