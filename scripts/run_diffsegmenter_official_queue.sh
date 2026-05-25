#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/tianyi/TF-OVCOS
REPO="$ROOT/third_party/official_methods/DiffSegmenter"
PY=/data/tianyi/conda_envs/tf-ovos/bin/python
LOG="$ROOT/runs/logs"
mkdir -p "$LOG" "$ROOT/runs/diffsegmenter"
cd "$REPO"
export HF_HOME=/data/tianyi/home_moved/.cache/huggingface
export TRANSFORMERS_CACHE=/data/tianyi/home_moved/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export DIFFSEG_DEVICE=${DIFFSEG_DEVICE:-cuda:0}
export DIFFSEG_SD_MODEL=${DIFFSEG_SD_MODEL:-runwayml/stable-diffusion-v1-5}
export DIFFSEG_BLIP_MODEL=${DIFFSEG_BLIP_MODEL:-Salesforce/blip-image-captioning-large}
export DIFFSEG_RESUME=1

# Official DiffSegmenter VOC12 (VOC20-style) run.
export DIFFSEG_OUTPUT="$ROOT/runs/diffsegmenter/voc12_wo_sy_norm"
export DIFFSEG_VOC12_ROOT="$ROOT/data/diffsegmenter_refs/VOCdevkit/VOC2012"
export DIFFSEG_VOC12_GT="$ROOT/data/diffsegmenter_refs/VOCdevkit/VOC2012/SegmentationClassAug"
$PY open_vocabulary/voc12/ptp_stable_best.py 2>&1 | tee "$LOG/diffsegmenter_voc12_predict.log"
$PY open_vocabulary/voc12/evaluation_voc12.py 2>&1 | tee "$LOG/diffsegmenter_voc12_eval.log"

# Official DiffSegmenter Pascal Context-59 run. This is official partial support, not COCO-Stuff/ADE.
export DIFFSEG_OUTPUT="$ROOT/runs/diffsegmenter/context59"
export DIFFSEG_VOC10_ROOT="$ROOT/data/diffsegmenter_refs/VOCdevkit/VOC2010"
export DIFFSEG_VOC10_GT="$ROOT/data/diffsegmenter_refs/VOCdevkit/VOC2010/SegmentationClassContext"
export DIFFSEG_VOC10_LIST="dataset/voc10/val.txt"
$PY open_vocabulary/voc10/ptp_stable_voc10.py 2>&1 | tee "$LOG/diffsegmenter_context59_predict.log"
$PY open_vocabulary/voc10/evaluation_voc10.py 2>&1 | tee "$LOG/diffsegmenter_context59_eval.log"
