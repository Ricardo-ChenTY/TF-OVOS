#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/tianyi/TF-OVCOS
FREE="$ROOT/third_party/official_methods/freeda"
SRC="$FREE/src"
ENV=/data/tianyi/conda_envs/freeda-official-py38-t112b
LOG="$ROOT/runs/logs/freeda_e1_queue.log"
GEN="$ROOT/runs/freeda/generated_configs"

mkdir -p "$ROOT/runs/logs" "$GEN"

log() { echo "[freeda-e1] $* $(date -Is)" | tee -a "$LOG"; }
run() { log "RUN $*"; "$@" 2>&1 | tee -a "$LOG"; }

write_eval_cfgs() {
  cat > "$GEN/eval_context59.yml" <<'YAML'
evaluate:
  pamr: false
  bg_thresh: 0.4
  kp_w: 0.3
  pred_qual_path: null
  gt_qual_path: null
  eval_only: true
  template: sub_imagenet_template
  task:
    - context59
  t_voc20: segmentation/configs/_base_/datasets/t_pascal_voc12_20.py
  t_context59: segmentation/configs/_base_/datasets/t_pascal_context59.py
  voc: segmentation/configs/_base_/datasets/pascal_voc12.py
  voc20: segmentation/configs/_base_/datasets/pascal_voc12_20.py
  context: segmentation/configs/_base_/datasets/pascal_context.py
  context59: segmentation/configs/_base_/datasets/pascal_context59.py
  coco_stuff: segmentation/configs/_base_/datasets/stuff.py
  coco_object: segmentation/configs/_base_/datasets/coco.py
  cityscapes: segmentation/configs/_base_/datasets/cityscapes.py
  ade20k: segmentation/configs/_base_/datasets/ade20k.py
YAML
}

check_ready() {
  test -x "$ENV/bin/python"
  test -f "$SRC/data/faiss_index/knn.index"
  test -d "$SRC/data/prototype_embeddings"
  test -e "$SRC/data/VOCdevkit/VOC2012"
  test -e "$SRC/data/VOCdevkit/VOC2010"
  test -e "$SRC/data/ade/ADEChallengeData2016"
  test -e "$SRC/data/coco_stuff164k/images"
}

metrics_done() {
  local out="$1"
  test -f "$out/log.txt" && rg -q "mIoU of .* test images|val/.*_miou" "$out/log.txt"
}

run_eval() {
  local name="$1"
  local eval_cfg="$2"
  local eval_base_cfg="$3"
  local out="$ROOT/runs/freeda/$name"

  if metrics_done "$out"; then
    log "skip $name; already complete"
    return
  fi

  cd "$SRC"
  log "starting $name"
  run "$ENV/bin/python" -m torch.distributed.run --nproc_per_node=1 main.py \
    --eval \
    --eval_cfg "$eval_cfg" \
    --eval_base_cfg "$eval_base_cfg" \
    --output "$out"
  log "finished $name"
}

main() {
  log "queue started"
  write_eval_cfgs
  check_ready

  run_eval voc20 configs/pascal20/freeda_pascal20.yml configs/pascal20/eval_pascal20.yml
  run_eval context59 configs/pascal59/freeda_pascal59.yml "$GEN/eval_context59.yml"
  run_eval ade20k150 configs/ade/freeda_ade.yml configs/ade/eval_ade.yml
  run_eval coco_stuff171 configs/cocostuff/freeda_cocostuff.yml configs/cocostuff/eval_cocostuff.yml

  log "queue finished"
}

main "$@"
