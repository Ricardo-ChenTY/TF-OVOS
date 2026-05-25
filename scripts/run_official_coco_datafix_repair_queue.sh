#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
RUNS="$ROOT/runs/official"
LOGS="$ROOT/runs/logs"

mkdir -p "$RUNS" "$LOGS"

"$PY" "$ROOT/scripts/prepare_official_mmseg_configs.py"

done_log() {
  local log="$1"
  local total="$2"
  [[ -f "$log" ]] && grep -Fq "Iter(test) [$total/$total]" "$log"
}

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    echo "[coco-repair] waiting for $session $(date -Is)"
    sleep 300
  done
}

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    echo "[coco-repair] waiting for current GPU job $(date -Is)"
    sleep 300
  done
}

run_coco() {
  local method="$1"
  local method_lower="$2"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/coco_stuff164k_datafix"
  local log="$LOGS/official_${method_lower}_coco_stuff164k_datafix.log"

  if done_log "$log" 5000; then
    echo "[coco-repair] skip $method coco_stuff164k; already complete"
    return
  fi

  wait_for_gpu_idle
  echo "[coco-repair] start $method coco_stuff164k $(date -Is)"
  (
    cd "$repo"
    "$PY" eval.py \
      --config "configs/cfg_tfovos_coco_stuff164k.py" \
      --work-dir "$work_dir"
  ) > "$log" 2>&1 || {
    echo "[coco-repair] failed $method coco_stuff164k; see $log"
    return 0
  }
  echo "[coco-repair] done $method coco_stuff164k $(date -Is)"
}

wait_for_session official_datafix_queue
run_coco SCLIP sclip
run_coco NACLIP naclip

echo "[coco-repair] official COCO datafix repair finished $(date -Is)"
