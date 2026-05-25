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
  [[ -f "$log" ]] && grep -Fq "Iter(test) [5000/5000]" "$log"
}

wait_if_running() {
  local work_dir="$1"
  while pgrep -f -- "--work-dir $work_dir" >/dev/null; do
    echo "[queue] waiting for active job: $work_dir $(date -Is)"
    sleep 300
  done
}

run_coco() {
  local method="$1"
  local method_lower="$2"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/coco_stuff164k_officialmap"
  local log="$LOGS/official_${method_lower}_coco_rerun_officialmap.log"

  wait_if_running "$work_dir"

  if done_log "$log"; then
    echo "[queue] skip $method coco_stuff164k_officialmap; already complete"
    return
  fi

  echo "[queue] start $method coco_stuff164k_officialmap $(date -Is)"
  cd "$repo"
  if [[ "$method" == "ResCLIP" ]]; then
    "$PY" eval.py \
      --config configs/cfg_tfovos_coco_stuff164k.py \
      --work-dir "$work_dir" \
      --arch wo_resi \
      --attn resclip \
      --std 5 \
      > "$log" 2>&1
  else
    "$PY" eval.py \
      --config configs/cfg_tfovos_coco_stuff164k.py \
      --work-dir "$work_dir" \
      > "$log" 2>&1
  fi
  echo "[queue] done $method coco_stuff164k_officialmap $(date -Is)"
}

run_coco SCLIP sclip
run_coco NACLIP naclip
run_coco ResCLIP resclip

echo "[queue] official COCO queue finished $(date -Is)"
