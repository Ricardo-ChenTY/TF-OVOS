#!/usr/bin/env bash
# Re-run proposal rows whose CLIP/SigLIP prompt vocabulary was not using official class names.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
LOGS="$ROOT/runs/logs"
mkdir -p "$LOGS"
cd "$ROOT"

log() { echo "[proposal-vocab-repair] $* $(date -Is)"; }

wait_for_session_to_finish() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for $session to finish before overwriting proposal outputs"
    sleep 300
  done
}

wait_for_gpu_idle() {
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt 35000 ]]; do
    log "waiting for GPU memory (need 35GB free)"
    sleep 60
  done
}

run_one() {
  local method="$1"
  local dataset="$2"
  local tag="$3"
  local logfile="$LOGS/proposal_official_vocab_${method}_${dataset}.log"
  wait_for_gpu_idle
  log "start $method $dataset with official vocab ($tag)"
  "$PY" -m tf_ovos.run_benchmark \
    --method "$method" \
    --dataset "${dataset}_val" \
    --num-shards 1 \
    2>&1 | tee "$logfile" || { log "FAILED $method $dataset; see $logfile"; return 0; }
  log "done $method $dataset"
}

wait_for_session_to_finish proposal_e1_queue

# Only re-run rows with a likely material prompt-vocabulary shift.
# Context459 changed 94 official class prompts and the old proposal scores looked abnormally low.
run_one sam_amg_clip context459 E2
run_one sam_amg_siglip context459 E2

# Do not spend the queue on tiny prompt deltas unless later analysis shows a >~1 pp effect:
# - context59: person -> people (single class-name difference)
# - ade20k150: 16 mostly spacing differences
# - ade20k847: one apostrophe difference

"$PY" "$ROOT/scripts/analyze_official_results.py" || true
"$PY" -m tf_ovos.summarize_results --run-root "$ROOT/runs" --out-dir "$ROOT/runs/tables" || true
log "official vocab repair queue finished"
