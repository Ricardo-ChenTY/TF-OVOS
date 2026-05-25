#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
RUNS="$ROOT/runs/official"
LOGS="$ROOT/runs/logs"

mkdir -p "$RUNS" "$LOGS"

"$PY" "$ROOT/scripts/prepare_official_e2_configs.py"

done_log() {
  local log="$1"
  local total="$2"
  [[ -f "$log" ]] || return 1
  if grep -Eq "Traceback|OutOfMemory|CUDA out of memory|failed|FAILED" "$log"; then
    return 1
  fi
  grep -Eq "Iter\\(test\\).*\\[ *$total/$total\\]" "$log" && \
    grep -Eq "aAcc: +[0-9.]+ +mIoU: +[0-9.]+ +mAcc:" "$log"
}

wait_for_gpu_idle() {
  local need_mb="${1:-50000}"
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt "$need_mb" ]]; do
    echo "[official-e2] waiting for GPU memory (need ${need_mb}MB free) $(date -Is)"
    sleep 60
  done
}

run_e2() {
  local method="$1"
  local method_lower="$2"
  local dataset="$3"
  local total="$4"
  local repo="$ROOT/third_party/official_methods/$method"
  local work_dir="$RUNS/$method_lower/${dataset}_e2"
  local log="$LOGS/official_${method_lower}_${dataset}_e2.log"

  if done_log "$log" "$total"; then
    echo "[official-e2] skip $method $dataset; already complete"
    return
  fi

  if [[ "$method" == "CorrCLIP" && "$dataset" == "ade847" ]]; then
    # This row OOMed only when launched alongside a ~60GB MaskCLIP job.
    # SAM alone is ~14GB, so 60GB free is enough and avoids blocking behind
    # the whole proposal queue.
    wait_for_gpu_idle 60000
  else
    wait_for_gpu_idle 50000
  fi
  echo "[official-e2] start $method $dataset $(date -Is)"
  (
    cd "$repo"
    if [[ "$method" == "CorrCLIP" ]]; then
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}_e2.py"
    elif [[ "$method" == "ResCLIP" ]]; then
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}_e2.py" \
        --work-dir "$work_dir" \
        --arch wo_resi \
        --attn resclip \
        --std 5
    else
      "$PY" eval.py \
        --config "configs/cfg_tfovos_${dataset}_e2.py" \
        --work-dir "$work_dir"
    fi
  ) > "$log" 2>&1 || {
    echo "[official-e2] failed $method $dataset; see $log"
    return 0
  }
  echo "[official-e2] done $method $dataset $(date -Is)"
}

# Generated configs wire the local official data tree and official class-name files.
# Dense CLIP rows were missed in the first E2 queue; keep them before the already-run
# Proxy/Corr rows so a repair run fills the missing large-vocab pairs and skips done logs.
run_e2 SCLIP sclip context459 5105
run_e2 SCLIP sclip ade847 2000
run_e2 NACLIP naclip context459 5105
run_e2 NACLIP naclip ade847 2000
run_e2 ResCLIP resclip context459 5105
run_e2 ResCLIP resclip ade847 2000
run_e2 ProxyCLIP proxyclip context459 5105
run_e2 ProxyCLIP proxyclip ade847 2000
run_e2 CorrCLIP corrclip context459 5105
run_e2 CorrCLIP corrclip ade847 2000

"$PY" "$ROOT/scripts/analyze_official_results.py"
echo "[official-e2] queue finished $(date -Is)"
