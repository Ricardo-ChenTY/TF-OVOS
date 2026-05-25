#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
RUNS="$ROOT/runs/official"
LOGS="$ROOT/runs/logs"

mkdir -p "$RUNS" "$LOGS"

log() { echo "[officialnames-voc20] $* $(date -Is)"; }

"$PY" "$ROOT/scripts/prepare_official_mmseg_configs.py"
"$PY" "$ROOT/scripts/prepare_official_voc20_officialnames_configs.py"

done_log() {
  local log_path="$1"
  [[ -f "$log_path" ]] && grep -Fq "Iter(test) [1449/1449]" "$log_path" && grep -Eq "mIoU|mAcc|aAcc" "$log_path"
}

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    log "waiting for current GPU jobs"
    sleep 300
  done
}

run_method() {
  local method="$1"
  local method_lower="$2"
  local repo="$ROOT/third_party/official_methods/$method"
  local config="configs/cfg_tfovos_voc20_officialnames.py"
  local work_dir="$RUNS/$method_lower/voc20_officialnames"
  local log_path="$LOGS/official_${method_lower}_voc20_officialnames.log"

  if done_log "$log_path"; then
    log "skip $method VOC20 officialnames; already complete"
    return
  fi

  wait_for_gpu_idle
  log "start $method VOC20 officialnames"
  (
    cd "$repo"
    if [[ "$method" == "ResCLIP" ]]; then
      "$PY" eval.py \
        --config "$config" \
        --work-dir "$work_dir" \
        --arch wo_resi \
        --attn resclip \
        --std 5
    elif [[ "$method" == "CorrCLIP" ]]; then
      # CorrCLIP's eval.py does not accept --work-dir; it sets work_dir from the config name.
      "$PY" eval.py \
        --config "$config"
    else
      "$PY" eval.py \
        --config "$config" \
        --work-dir "$work_dir"
    fi
  ) > "$log_path" 2>&1 || {
    log "FAILED $method VOC20 officialnames; see $log_path"
    return 0
  }
  log "done $method VOC20 officialnames"
}

run_method SCLIP sclip
run_method NACLIP naclip
run_method ResCLIP resclip
run_method ProxyCLIP proxyclip
run_method CorrCLIP corrclip

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
log "official VOC20 officialnames queue finished"
