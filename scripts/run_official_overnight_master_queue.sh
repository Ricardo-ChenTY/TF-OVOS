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
    echo "[overnight-master] waiting for current GPU job $(date -Is)"
    sleep 300
  done
}

run_mmseg() {
  local method="$1"
  local method_lower="$2"
  local dataset="$3"
  local total="$4"
  local suffix="$5"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/${dataset}_${suffix}"
  local log="$LOGS/official_${method_lower}_${dataset}_${suffix}.log"

  case "$method" in
    SCLIP|NACLIP|ResCLIP|ProxyCLIP|CorrCLIP) ;;
    *)
      echo "[overnight-master] refusing method outside proposal official queue: $method"
      return 1
      ;;
  esac

  if done_log "$log" "$total"; then
    echo "[overnight-master] skip $method $dataset $suffix; already complete"
    return
  fi

  wait_for_gpu_idle
  echo "[overnight-master] start $method $dataset $suffix $(date -Is)"
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
    echo "[overnight-master] failed $method $dataset $suffix; see $log"
    return 0
  }
  echo "[overnight-master] done $method $dataset $suffix $(date -Is)"
}

run_e1_method() {
  local method="$1"
  local method_lower="$2"
  run_mmseg "$method" "$method_lower" voc20 1449 e1
  run_mmseg "$method" "$method_lower" context59 5105 e1
  run_mmseg "$method" "$method_lower" ade20k 2000 e1
  run_mmseg "$method" "$method_lower" coco_stuff164k 5000 e1
}

# Re-run COCO for CLIP-dense official repos with each method's official scale.
# Previous "officialmap" COCO logs used a shared 448 height for NACLIP/ResCLIP,
# so keep those only as high-resolution diagnostics.
run_mmseg SCLIP sclip coco_stuff164k 5000 officialscale
run_mmseg NACLIP naclip coco_stuff164k 5000 officialscale
run_mmseg ResCLIP resclip coco_stuff164k 5000 officialscale

# Fill proposal-table official methods that have runnable official repos here.
run_e1_method ProxyCLIP proxyclip
run_e1_method CorrCLIP corrclip

echo "[overnight-master] official overnight queue finished $(date -Is)"
