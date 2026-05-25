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

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    echo "[proposal-e1] waiting for current official GPU job $(date -Is)"
    sleep 300
  done
}

run_mmseg_method() {
  local method="$1"
  local method_lower="$2"
  local dataset="$3"
  local total="$4"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/$dataset"
  local log="$LOGS/official_${method_lower}_${dataset}.log"

  case "$method" in
    SCLIP|NACLIP|ResCLIP|ProxyCLIP|CorrCLIP) ;;
    *)
      echo "[proposal-e1] refusing method outside proposal official queue: $method"
      return 1
      ;;
  esac

  if done_log "$log" "$total"; then
    echo "[proposal-e1] skip $method $dataset; already complete"
    return
  fi

  wait_for_gpu_idle
  echo "[proposal-e1] start $method $dataset $(date -Is)"
  (
    cd "$repo"
    if [[ "$method" == "ResCLIP" ]]; then
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}.py" \
        --work-dir "$work_dir" \
        --arch wo_resi \
        --attn resclip \
        --std 5
    elif [[ "$method" == "CorrCLIP" ]]; then
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}.py"
    else
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}.py" \
        --work-dir "$work_dir"
    fi
  ) > "$log" 2>&1 || {
    echo "[proposal-e1] failed $method $dataset; see $log"
    return 0
  }
  echo "[proposal-e1] done $method $dataset $(date -Is)"
}

run_e1_method() {
  local method="$1"
  local method_lower="$2"
  run_mmseg_method "$method" "$method_lower" voc20 1449
  run_mmseg_method "$method" "$method_lower" context59 5105
  run_mmseg_method "$method" "$method_lower" ade20k 2000
  run_mmseg_method "$method" "$method_lower" coco_stuff164k 5000
}

# SCLIP/NACLIP/ResCLIP are handled by the existing official phase-1/COCO
# rerun logs. This queue only fills proposal-table methods that still need
# official-repo E1 runs from scratch.
run_e1_method ProxyCLIP proxyclip
run_e1_method CorrCLIP corrclip

echo "[proposal-e1] official E1 queue finished $(date -Is)"
