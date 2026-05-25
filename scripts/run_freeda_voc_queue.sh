#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/tianyi/TF-OVCOS
FREE="$ROOT/third_party/official_methods/freeda"
SRC="$FREE/src"
ENV=/data/tianyi/conda_envs/freeda-official-py38-t112b
CONDA=/data/tianyi/miniconda3/bin/conda
LOG="$ROOT/runs/logs/freeda_voc_queue.log"

mkdir -p "$ROOT/runs/logs"
log() { echo "[freeda-voc] $* $(date -Is)" | tee -a "$LOG"; }
run() { log "RUN $*"; "$@" 2>&1 | tee -a "$LOG"; }

check_artifacts() {
  test -f "$SRC/data/faiss_index/knn.index"
  test -d "$SRC/data/prototype_embeddings"
  test -e "$SRC/data/VOCdevkit"
}

ensure_env() {
  if [[ ! -x "$ENV/bin/python" ]]; then
    log "creating FreeDA official env at $ENV"
    run "$CONDA" create -y -p "$ENV" python=3.8
    run "$ENV/bin/python" -m pip install -U pip setuptools wheel
    run "$ENV/bin/python" -m pip install \
      torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
      --extra-index-url https://download.pytorch.org/whl/cu113
    run "$ENV/bin/python" -m pip install \
      mmcv-full==1.6.2 \
      -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
    : "requirements are installed below for both fresh and existing envs"
  else
    log "FreeDA env already exists at $ENV"
  fi

  grep -v -E "^(mysqlclient==|Pattern==)" "$SRC/requirements.txt" > "$ROOT/runs/logs/freeda_requirements_infer.txt"
  run "$ENV/bin/python" -m pip install Cython==0.29.36
  run "$ENV/bin/python" -m pip install --no-build-isolation pycocotools==2.0.6
  run "$ENV/bin/python" -m pip install -r "$ROOT/runs/logs/freeda_requirements_infer.txt"

  run "$ENV/bin/python" - <<'PY'
import torch
import mmcv
import mmseg
import faiss
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("mmcv", mmcv.__version__, "mmseg", mmseg.__version__)
print("faiss ok")
PY
}

run_voc20() {
  cd "$SRC"
  log "starting FreeDA official VOC20 eval"
  run "$ENV/bin/python" -m torch.distributed.run --nproc_per_node=1 main.py \
    --eval \
    --eval_cfg configs/pascal20/freeda_pascal20.yml \
    --eval_base_cfg configs/pascal20/eval_pascal20.yml \
    --output "$ROOT/runs/freeda/voc20"
  log "FreeDA VOC20 eval finished"
}

main() {
  log "queue started"
  check_artifacts
  ensure_env
  run_voc20
}

main "$@"
