#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
REPO="$ROOT/third_party/official_methods/maskclip"
RUNS="$ROOT/runs/official/maskclip"
LOGS="$ROOT/runs/logs"
CKPT="pretrain/ViT16_clip_backbone.pth"

mkdir -p "$RUNS" "$LOGS"

log() { echo "[official-maskclip] $* $(date -Is)"; }

wait_for_tmux_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; do
    log "waiting for current GPU jobs"
    sleep 300
  done
}

done_workdir() {
  local work_dir="$1"
  find "$work_dir" -maxdepth 1 -type f -name 'eval_single_scale_*.json' 2>/dev/null | grep -q .
}

run_dataset() {
  local tag="$1"
  local config="$2"
  local work_dir="$RUNS/$tag"
  local log_path="$LOGS/official_maskclip_${tag}.log"

  if done_workdir "$work_dir"; then
    log "skip $tag; already complete"
    return
  fi

  wait_for_gpu_idle
  log "start $tag"
  (
    cd "$REPO"
    "$PY" "$ROOT/scripts/run_maskclip_official_test.py" \
      "$config" \
      "$CKPT" \
      --eval mIoU \
      --work-dir "$work_dir"
  ) > "$log_path" 2>&1 || {
    log "FAILED $tag; see $log_path"
    return 0
  }
  log "done $tag"
}

"$PY" "$ROOT/scripts/prepare_official_mmseg_configs.py"
"$PY" "$ROOT/scripts/prepare_official_maskclip_configs.py"

# Keep already scheduled VOC official-name repair and proposal VOC repair ahead
# of this queue; proposal_e1_queue is patched to wait for this session later.
wait_for_tmux_done official_voc20_officialnames_queue
wait_for_tmux_done proposal_voc20_officialnames_queue
wait_for_tmux_done official_voc20_repair_queue

run_dataset voc20 configs/tfovos/maskclip_vit16_tfovos_voc20.py
run_dataset context59 configs/tfovos/maskclip_vit16_tfovos_context59.py
run_dataset ade150 configs/tfovos/maskclip_vit16_tfovos_ade150.py
run_dataset coco171 configs/tfovos/maskclip_vit16_tfovos_coco171.py

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
log "official MaskCLIP queue finished"
