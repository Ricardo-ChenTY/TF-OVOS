#!/usr/bin/env bash
# Strict official OVDiff VOC queue: build isolated official env, then run the README pipeline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVDIFF="$ROOT/third_party/official_methods/ovdiff"
CONDA="/data/tianyi/miniconda3/bin/conda"
ENV="/data/tianyi/conda_envs/ovdiff-official"
LOGS="$ROOT/runs/logs"
LOG="$LOGS/ovdiff_voc_queue_driver.log"
mkdir -p "$LOGS"

log() { echo "[ovdiff-voc] $* $(date -Is)" | tee -a "$LOG"; }
run() { log "RUN $*"; "$@" 2>&1 | tee -a "$LOG"; }

prepare_data() {
  log "preparing OVDiff VOC data links"
  mkdir -p "$OVDIFF/data/PascalVOC"
  if [[ ! -e "$OVDIFF/data/PascalVOC/VOCdevkit" && ! -L "$OVDIFF/data/PascalVOC/VOCdevkit" ]]; then
    ln -s "$ROOT/data/raw/VOCdevkit" "$OVDIFF/data/PascalVOC/VOCdevkit"
  fi
  if [[ ! -e "$OVDIFF/data/VOCdevkit" && ! -L "$OVDIFF/data/VOCdevkit" ]]; then
    ln -s "$ROOT/data/raw/VOCdevkit" "$OVDIFF/data/VOCdevkit"
  fi
  test -f "$OVDIFF/CutLER/cutler_cascade_final.pth"
}

ensure_env() {
  if [[ ! -x "$ENV/bin/python" ]]; then
    log "creating OVDiff official conda env at $ENV"
    run "$CONDA" env create --prefix "$ENV" --file "$OVDIFF/conda_environment.yml"
  else
    log "OVDiff env already exists at $ENV"
  fi
  run "$ENV/bin/python" - <<'PYCHECK'
import torch, diffusers, transformers, timm
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'timm', timm.__version__)
try:
    import mmcv, mmseg
    print('mmcv', mmcv.__version__, 'mmseg', mmseg.__version__)
except Exception as exc:
    print('mmcv/mmseg import failed:', repr(exc))
try:
    import detectron2
    print('detectron2 ok')
except Exception as exc:
    print('detectron2 import failed:', repr(exc))
PYCHECK
}

run_pipeline() {
  cd "$OVDIFF"
  mkdir -p outputs/voc outputs/runs/voc

  log "starting OVDiff official VOC pipeline"
  run "$ENV/bin/python" sample_support_set.py voc outputs/voc
  run "$ENV/bin/python" gen_vit_features.py --model_key clip_ViT-B/16 --layer -2 voc outputs/voc
  run "$ENV/bin/python" gen_vit_features.py --model_key dino_vitb8 voc outputs/voc
  run "$ENV/bin/python" gen_proto_vit.py voc outputs/voc --feature_path_prefix dino/dino_vitb8_8_0 dino_vitb8_cfbgv3_bpp_k32_n32_s43_off0
  run "$ENV/bin/python" gen_proto_vit.py voc outputs/voc --feature_path_prefix clip/clip_vit-b_16_16_-2_0 clipb16_-2_cfbgv3_bpp_k32_n32_s43_off0
  run "$ENV/bin/python" gen_proto_sd.py voc outputs/voc sd_k32_n32_s43_off0
  run "/bin/python" predict.py \
    voc outputs/runs/voc \
    --prots outputs/voc/{dataset}_sd_k32_n32_s43_off0_0,6:13,15+_t200_proto.pt \
            outputs/voc/{dataset}_clipb16_-2_cfbgv3_bpp_k32_n32_s43_off0_proto.pt \
            outputs/voc/{dataset}_dino_vitb8_cfbgv3_bpp_k32_n32_s43_off0_proto.pt
  log "OVDiff VOC pipeline finished"
}

main() {
  log "queue started"
  prepare_data
  ensure_env
  run_pipeline
}

main "$@"
