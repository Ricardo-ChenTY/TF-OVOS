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

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    echo "[coco-canonical] waiting for current GPU job $(date -Is)"
    sleep 300
  done
}

run_coco() {
  local method="$1"
  local method_lower="$2"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/coco_stuff164k_officialscale"
  local log="$LOGS/official_${method_lower}_coco_rerun_officialscale.log"

  case "$method" in
    SCLIP|NACLIP|ResCLIP|ProxyCLIP|CorrCLIP) ;;
    *)
      echo "[coco-canonical] refusing non-proposal method: $method"
      return 1
      ;;
  esac

  if done_log "$log"; then
    echo "[coco-canonical] skip $method coco; already complete"
    return
  fi

  wait_for_gpu_idle
  echo "[coco-canonical] start $method coco $(date -Is)"
  (
    cd "$repo"
    if [[ "$method" == "ResCLIP" ]]; then
      "$PY" eval.py \
        --config configs/cfg_tfovos_coco_stuff164k.py \
        --work-dir "$work_dir" \
        --arch wo_resi \
        --attn resclip \
        --std 5
    elif [[ "$method" == "CorrCLIP" ]]; then
      "$PY" eval.py \
        --config configs/cfg_tfovos_coco_stuff164k.py
    else
      "$PY" eval.py \
        --config configs/cfg_tfovos_coco_stuff164k.py \
        --work-dir "$work_dir"
    fi
  ) > "$log" 2>&1 || {
    echo "[coco-canonical] failed $method coco; see $log"
    return 0
  }
  echo "[coco-canonical] done $method coco $(date -Is)"
}

run_coco SCLIP sclip
run_coco NACLIP naclip
run_coco ResCLIP resclip
run_coco ProxyCLIP proxyclip
run_coco CorrCLIP corrclip

echo "[coco-canonical] queue finished $(date -Is)"
