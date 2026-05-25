#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/tianyi/TF-OVCOS
ENV=/data/tianyi/conda_envs/ovdiff-official
LOG="$ROOT/runs/logs/ovdiff_env_repair2.log"

mkdir -p "$ROOT/runs/logs"
{
  echo "[ovdiff-env-repair2] start $(date -Is)"
  "$ENV/bin/python" -m pip install "setuptools==65.6.3" "wheel==0.38.4"
  "$ENV/bin/python" -m pip install "huggingface_hub==0.13.4"
  "$ENV/bin/python" -c "import torch, mmcv, mmseg, diffusers, transformers, timm, clip; print('core OK', torch.__version__, mmcv.__version__, mmseg.__version__)"
  if ! "$ENV/bin/python" -c "import detectron2; print('detectron2 OK', detectron2.__version__)"; then
    echo "[ovdiff-env-repair2] installing detectron2 $(date -Is)"
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++
    export CFLAGS="-include cstdint"
    export CXXFLAGS="-include cstdint"
    export MAX_JOBS=8
    "$ENV/bin/python" -m pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git@v0.6"
  fi
  "$ENV/bin/python" -c "import detectron2; print('detectron2 final OK', detectron2.__version__)"
  echo "[ovdiff-env-repair2] done $(date -Is)"
} >> "$LOG" 2>&1
