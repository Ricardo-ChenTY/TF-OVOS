#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"

mkdir -p "$LOGS"

"$PY" "$ROOT/scripts/prepare_official_mmseg_configs.py"

done_log() {
  local log="$1"
  local total="$2"
  [[ -f "$log" ]] && grep -Fq "Iter(test) [$total/$total]" "$log"
}

wait_for_datafix_queue() {
  while tmux has-session -t official_datafix_queue 2>/dev/null; do
    echo "[corrclip-datafix] waiting for official_datafix_queue $(date -Is)"
    sleep 300
  done
  while tmux has-session -t official_coco_datafix_repair_queue 2>/dev/null; do
    echo "[corrclip-datafix] waiting for official_coco_datafix_repair_queue $(date -Is)"
    sleep 300
  done
}

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    echo "[corrclip-datafix] waiting for current GPU job $(date -Is)"
    sleep 300
  done
}

run_corrclip() {
  local dataset="$1"
  local total="$2"
  local repo="$ROOT/third_party/official_methods/CorrCLIP"
  local log="$LOGS/official_corrclip_${dataset}_datafix.log"

  if done_log "$log" "$total"; then
    echo "[corrclip-datafix] skip CorrCLIP $dataset; already complete"
    return
  fi

  wait_for_datafix_queue
  wait_for_gpu_idle
  echo "[corrclip-datafix] start CorrCLIP $dataset $(date -Is)"
  (
    cd "$repo"
    "$PY" eval.py --config "configs/cfg_tfovos_${dataset}.py"
  ) > "$log" 2>&1 || {
    echo "[corrclip-datafix] failed CorrCLIP $dataset; see $log"
    return 0
  }
  echo "[corrclip-datafix] done CorrCLIP $dataset $(date -Is)"
}

run_corrclip voc20 1449
run_corrclip context59 5105
run_corrclip ade20k 2000
run_corrclip coco_stuff164k 5000

echo "[corrclip-datafix] official CorrCLIP datafix queue finished $(date -Is)"
