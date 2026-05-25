#!/usr/bin/env bash
# Run Trident and CASS under the shared TF-OVOS protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
RUNS="$ROOT/runs/official"

mkdir -p "$LOGS" "$RUNS"

log() { echo "[trident-cass] $* $(date -Is)"; }

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session: $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  local need_mb="${1:-50000}"
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

run_method_dataset() {
  local method="$1"
  local repo="$2"
  local dataset="$3"
  local total="$4"
  local config="configs/cfg_tfovos_${dataset}.py"
  local work_dir="$RUNS/${method}/${dataset}"
  local log_file="$LOGS/official_${method}_${dataset}.log"

  if done_log "$log_file" "$total"; then
    log "skip $method $dataset; already complete"
    return
  fi

  wait_for_gpu_idle 50000
  log "start $method $dataset"
  (
    cd "$repo"
    if [[ "$method" == "cass" ]]; then
      "$PY" eval.py --config "$config" --pamr off --work-dir "$work_dir"
    else
      "$PY" eval.py --config "$config" --work-dir "$work_dir"
    fi
  ) > "$log_file" 2>&1 || {
    log "FAILED $method $dataset; see $log_file"
    return 0
  }
  log "done $method $dataset"
}

"$PY" "$ROOT/scripts/prepare_trident_cass_configs.py"

wait_for_session odise_partial_fixed
wait_for_session ovseg_context59_fix_after_odise
wait_for_session official_dense_e2_after_refs
wait_for_session dinov2_sam_after_e2

TRIDENT="$ROOT/third_party/official_methods/Trident"
CASS="$ROOT/third_party/official_methods/CASS"

for dataset in voc20 context59 ade20k coco_stuff164k context459 ade847; do
  case "$dataset" in
    voc20) total=1449 ;;
    context59) total=5105 ;;
    ade20k) total=2000 ;;
    coco_stuff164k) total=5000 ;;
    context459) total=5105 ;;
    ade847) total=2000 ;;
  esac
  run_method_dataset trident "$TRIDENT" "$dataset" "$total"
done

for dataset in voc20 context59 ade20k coco_stuff164k context459 ade847; do
  case "$dataset" in
    voc20) total=1449 ;;
    context59) total=5105 ;;
    ade20k) total=2000 ;;
    coco_stuff164k) total=5000 ;;
    context459) total=5105 ;;
    ade847) total=2000 ;;
  esac
  run_method_dataset cass "$CASS" "$dataset" "$total"
done

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
log "Trident/CASS queue finished"
