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
    echo "[datafix] waiting for current GPU job $(date -Is)"
    sleep 300
  done
}

run_mmseg() {
  local method="$1"
  local method_lower="$2"
  local dataset="$3"
  local total="$4"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/${dataset}_datafix"
  local log="$LOGS/official_${method_lower}_${dataset}_datafix.log"

  case "$method" in
    SCLIP|NACLIP|ResCLIP|ProxyCLIP) ;;
    *)
      echo "[datafix] refusing method outside datafix queue: $method"
      return 1
      ;;
  esac

  if done_log "$log" "$total"; then
    echo "[datafix] skip $method $dataset; already complete"
    return
  fi

  wait_for_gpu_idle
  echo "[datafix] start $method $dataset $(date -Is)"
  (
    cd "$repo"
    if [[ "$method" == "ResCLIP" ]]; then
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}.py" \
        --work-dir "$work_dir" \
        --arch wo_resi \
        --attn resclip \
        --std 5
    else
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}.py" \
        --work-dir "$work_dir"
    fi
  ) > "$log" 2>&1 || {
    echo "[datafix] failed $method $dataset; see $log"
    return 0
  }
  echo "[datafix] done $method $dataset $(date -Is)"
}

run_e1_method() {
  local method="$1"
  local method_lower="$2"
  run_mmseg "$method" "$method_lower" voc20 1449
  run_mmseg "$method" "$method_lower" context59 5105
  run_mmseg "$method" "$method_lower" ade20k 2000
  run_mmseg "$method" "$method_lower" coco_stuff164k 5000
}

run_e1_method SCLIP sclip
run_e1_method NACLIP naclip
run_e1_method ResCLIP resclip
run_e1_method ProxyCLIP proxyclip

echo "[datafix] official datafix queue finished $(date -Is)"
