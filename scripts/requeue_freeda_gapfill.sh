#!/bin/bash
# Wait for the existing ADE-847 solo-requeue wrapper (naclip/scclip/cass/
# trident) to finish, then do a full solo re-run of FreeDA on both datasets
# (its earlier concurrent run only hit 99.4%/93.5% completion due to OOM
# under GPU contention -- this gap-fills it to match the other 5 methods'
# ~100% coverage). Uses FreeDA's own py38 conda env and an explicit
# absolute output path (the earlier run's relative --output-csv landed
# inside third_party/official_methods/freeda/src/ because the script
# internally chdirs there).
set -uo pipefail
cd /data/tianyi/TF-OVCOS
FREEDA_PY=/data/tianyi/conda_envs/freeda-official-py38-t112b/bin/python

echo "[$(date)] waiting for requeue_ade847_solo.sh (PID 505746) to finish..."
while kill -0 505746 2>/dev/null; do
  sleep 60
done
echo "[$(date)] ADE-847 solo requeue wrapper finished. Starting FreeDA full gap-fill re-run."

$FREEDA_PY scripts/real_obs5_diagnostic_freeda.py \
  --datasets context459 ade847 \
  --num-images 5105 \
  --output-csv /data/tianyi/TF-OVCOS/runs/analysis/obs5_real_diagnostic_freeda_full_gapfill.csv \
  > runs/logs/obs5_freeda_gapfill.log 2>&1

echo "[$(date)] FreeDA gap-fill re-run finished (exit=$?)."
