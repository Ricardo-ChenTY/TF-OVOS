#!/usr/bin/env bash
# Run Phase 3 proposal adapter E1+E2 evaluation queue.
# Waits for SAM weights download, then runs SAM-AMG+CLIP and SAM-AMG+SigLIP.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
WEIGHTS="$ROOT/weights/sam_vit_h_4b8939.pth"

mkdir -p "$LOGS"

log() { echo "[proposal-e1] $* $(date -Is)"; }

wait_for_weights() {
  while [[ ! -f "$WEIGHTS" ]]; do
    log "waiting for SAM weights at $WEIGHTS ..."
    sleep 60
  done
  # Wait until file is not being written (size stable for 30s)
  local prev_size=0
  while true; do
    local cur_size
    cur_size=$(stat -c%s "$WEIGHTS" 2>/dev/null || echo 0)
    if [[ "$cur_size" -eq "$prev_size" && "$cur_size" -gt 0 ]]; then
      break
    fi
    prev_size="$cur_size"
    sleep 30
  done
  log "SAM weights ready ($(du -sh "$WEIGHTS" | cut -f1))"
}

wait_for_gpu_idle() {
  # SAM ViT-H + VLM needs ~25GB; wait only if less than 35GB free
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt 35000 ]]; do
    log "waiting for GPU memory (need 35GB free)..."
    sleep 60
  done
}

wait_for_priority_queues() {
  local queues=(official_voc20_officialnames_queue proposal_voc20_officialnames_queue official_maskclip_queue)
  while true; do
    local active=()
    for q in "${queues[@]}"; do
      if tmux has-session -t "$q" 2>/dev/null; then
        active+=("$q")
      fi
    done
    if [[ "${#active[@]}" -eq 0 ]]; then
      break
    fi
    log "waiting for priority queue(s): ${active[*]}"
    sleep 300
  done
}

metrics_done() {
  local method="$1"
  local dataset="$2"
  local metrics="$ROOT/runs/${method}/${dataset}_val/metrics.json"
  [[ -f "$metrics" ]] && grep -qF '"miou"' "$metrics"
}

done_log() {
  local log="$1"
  local total="$2"
  [[ -f "$log" ]] && grep -Fq "\"progress\": 100" "$log" || \
  [[ -f "$log" ]] && grep -Fq "mIoU" "$log"
}

run_adapter() {
  local method="$1"
  local dataset="$2"
  local log="$LOGS/proposal_${method}_${dataset}_e1.log"

  if metrics_done "$method" "$dataset"; then
    log "skip $method $dataset; already complete"
    return
  fi

  wait_for_priority_queues
  wait_for_gpu_idle
  log "start $method $dataset"
  "$PY" -m tf_ovos.run_benchmark \
    --method "$method" \
    --dataset "${dataset}_val" \
    --num-shards 1 \
    --skip-existing \
    2>&1 | tee "$log" || {
    log "FAILED $method $dataset; see $log"
    return 0
  }
  log "done $method $dataset"
}

run_adapter_e2() {
  local method="$1"
  local log_ctx="$LOGS/proposal_${method}_context459_e2.log"
  local log_ade="$LOGS/proposal_${method}_ade847_e2.log"

  if ! metrics_done "$method" context459; then
    wait_for_priority_queues
    wait_for_gpu_idle
    log "start $method context459 E2"
    "$PY" -m tf_ovos.run_benchmark \
      --method "$method" \
      --dataset context459_val \
      --num-shards 1 \
      --skip-existing \
      2>&1 | tee "$log_ctx" || { log "FAILED $method context459 E2"; }
  else
    log "skip $method context459 E2; already complete"
  fi

  if ! metrics_done "$method" ade20k847; then
    wait_for_priority_queues
    wait_for_gpu_idle
    log "start $method ade847 E2"
    "$PY" -m tf_ovos.run_benchmark \
      --method "$method" \
      --dataset ade20k847_val \
      --num-shards 1 \
      --skip-existing \
      2>&1 | tee "$log_ade" || { log "FAILED $method ade847 E2"; }
  else
    log "skip $method ade847 E2; already complete"
  fi
}

# ── Main queue ──────────────────────────────────────────────────────────────

wait_for_weights

# E1: standard 4 datasets for each adapter
for method in sam_amg_clip sam_amg_siglip; do
  run_adapter "$method" voc20
  run_adapter "$method" context59
  run_adapter "$method" ade20k150
  run_adapter "$method" coco_stuff171
done

# E2: large-vocab for each adapter
for method in sam_amg_clip sam_amg_siglip; do
  run_adapter_e2 "$method"
done

# Refresh analysis
"$PY" "$ROOT/scripts/analyze_official_results.py" || true
"$PY" -m tf_ovos.summarize_results --run-root "$ROOT/runs" --out-dir "$ROOT/runs/tables" || true

log "proposal E1+E2 queue finished"
