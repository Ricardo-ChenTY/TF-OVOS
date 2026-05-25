#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/tianyi/TF-OVCOS
ENV=/data/tianyi/conda_envs/ovdiff-official
LOG="$ROOT/runs/logs/ovdiff_sd15_download.log"

mkdir -p "$ROOT/runs/logs"
{
  echo "[ovdiff-sd15] start $(date -Is)"
  "$ENV/bin/python" - <<'PY'
import torch
from diffusers import StableDiffusionPipeline

StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    local_files_only=False,
)
print("download ok")
PY
  echo "[ovdiff-sd15] done $(date -Is)"
} >> "$LOG" 2>&1
