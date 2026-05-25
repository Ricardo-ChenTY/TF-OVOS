#!/usr/bin/env bash
# Fill remaining CLIPtrase and FreeDA E2 gaps after the main proposal queues.
set -euo pipefail

ROOT=/data/tianyi/TF-OVCOS
PY=/data/tianyi/conda_envs/tf-ovos/bin/python
FREE="$ROOT/third_party/official_methods/freeda"
SRC="$FREE/src"
FREE_ENV=/data/tianyi/conda_envs/freeda-official-py38-t112b
LOG="$ROOT/runs/logs/freeda_cliptrase_repair_queue.log"
GEN="$ROOT/runs/freeda/generated_configs"

mkdir -p "$ROOT/runs/logs" "$GEN"

log() { echo "[freeda-cliptrase-repair] $* $(date -Is)" | tee -a "$LOG"; }
run() { log "RUN $*"; "$@" 2>&1 | tee -a "$LOG"; }

wait_for_session() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session: $session"
    sleep 300
  done
}

wait_for_gpu_idle() {
  local need_mb="${1:-45000}"
  while [[ $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) -lt "$need_mb" ]]; do
    log "waiting for GPU memory (need ${need_mb}MB free)..."
    sleep 120
  done
}

write_freeda_e2_eval_cfgs() {
  cat > "$GEN/eval_context459.yml" <<'YAML'
evaluate:
  pamr: false
  bg_thresh: 0.4
  kp_w: 0.3
  pred_qual_path: null
  gt_qual_path: null
  eval_only: true
  template: sub_imagenet_template
  task:
    - context459
  context459: segmentation/configs/_base_/datasets/pascal_context459.py
YAML

  cat > "$GEN/eval_ade847.yml" <<'YAML'
evaluate:
  pamr: false
  bg_thresh: 0.4
  kp_w: 0.3
  pred_qual_path: null
  gt_qual_path: null
  eval_only: true
  template: sub_imagenet_template
  task:
    - ade847
  ade847: segmentation/configs/_base_/datasets/ade20k847.py
YAML
}

freeda_ready() {
  test -x "$FREE_ENV/bin/python"
  test -f "$SRC/data/faiss_index/knn.index"
  test -d "$SRC/data/prototype_embeddings"
  test -e "$ROOT/data/detectron2_refs/pcontext_full/val/label"
  test -e "$ROOT/data/detectron2_refs/ADE20K_2021_17_01/images_detectron2/validation"
  test -e "$ROOT/data/detectron2_refs/ADE20K_2021_17_01/annotations_detectron2/validation"
}

freeda_done() {
  local out="$1"
  test -f "$out/log.txt" && rg -q "mIoU of .* test images|val/.*_miou" "$out/log.txt"
}

run_freeda_eval() {
  local name="$1"
  local eval_cfg="$2"
  local eval_base_cfg="$3"
  local out="$ROOT/runs/freeda/$name"

  if freeda_done "$out"; then
    log "skip FreeDA $name; already complete"
    return
  fi

  wait_for_gpu_idle 45000
  cd "$SRC"
  log "starting FreeDA $name"
  run "$FREE_ENV/bin/python" -m torch.distributed.run --nproc_per_node=1 main.py \
    --eval \
    --eval_cfg "$eval_cfg" \
    --eval_base_cfg "$eval_base_cfg" \
    --output "$out"
  log "finished FreeDA $name"
}

main() {
  log "queue started"
  wait_for_session scclip_after_trident

  log "running CLIPtrase remaining E1/E2 gaps"
  bash "$ROOT/scripts/run_official_cliptrase_queue.sh" 2>&1 | tee -a "$LOG" || true

  write_freeda_e2_eval_cfgs
  freeda_ready

  run_freeda_eval context459 configs/pascal59/freeda_pascal59.yml "$GEN/eval_context459.yml"
  run_freeda_eval ade847 configs/ade/freeda_ade.yml "$GEN/eval_ade847.yml"

  "$PY" "$ROOT/scripts/analyze_official_results.py" 2>&1 | tee -a "$LOG" || true
  log "queue finished"
}

main "$@"
