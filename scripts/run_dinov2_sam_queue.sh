#!/usr/bin/env bash
# Run implemented DINOv2+SAM proposal adapters after higher-priority queues.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
WEIGHTS="$ROOT/weights/sam_vit_h_4b8939.pth"

mkdir -p "$LOGS"

log() { echo "[dinov2-sam] $* $(date -Is)"; }

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session: $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt 45000 ]]; do
    log "waiting for GPU memory (need 45GB free for SAM + VLM + DINOv2)..."
    sleep 120
  done
}

metrics_done() {
  local method="$1"
  local dataset="$2"
  local metrics="$ROOT/runs/${method}/${dataset}_val/metrics.json"
  [[ -f "$metrics" ]] && grep -qF '"miou"' "$metrics"
}

run_dataset() {
  local method="$1"
  local dataset="$2"
  local log_file="$LOGS/dinov2_sam_${method}_${dataset}.log"

  if metrics_done "$method" "$dataset"; then
    log "skip $method $dataset; already complete"
    return
  fi

  wait_for_gpu_idle
  log "start $method $dataset"
  "$PY" -m tf_ovos.run_benchmark \
    --method "$method" \
    --dataset "${dataset}_val" \
    --num-shards 1 \
    --skip-existing \
    2>&1 | tee "$log_file" || {
      log "FAILED $method $dataset; see $log_file"
      return 0
    }
  log "done $method $dataset"
}

wait_for_session odise_partial_fixed
wait_for_session ovseg_context59_fix_after_odise
wait_for_session official_dense_e2_after_refs

if [[ ! -f "$WEIGHTS" ]]; then
  log "SAM checkpoint missing: $WEIGHTS"
  exit 1
fi

for method in dinov2_sam_siglip dinov2_sam_clip; do
  for dataset in voc20 context59 ade20k150 coco_stuff171 context459 ade20k847; do
    run_dataset "$method" "$dataset"
  done
done

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
"$PY" -m tf_ovos.summarize_results --run-root "$ROOT/runs" --out-dir "$ROOT/runs/tables" || true

log "DINOv2+SAM queue finished"
