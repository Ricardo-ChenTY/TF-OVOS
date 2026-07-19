#!/bin/bash
# Wait for sam_amg_siglip's full run to finish, then serially (solo GPU,
# no contention) re-run ADE-847 only for naclip/scclip/cass/trident to
# recover full coverage (their concurrent run only completed 58-77% of
# ADE-847 due to OOM under 4-way GPU contention). Context-459 is already
# ~100% complete for all 4 and is left untouched. Output goes to separate
# "_ade847_solo" CSVs so the existing full CSVs (with their Context-459
# rows) are not overwritten.
set -uo pipefail
cd /data/tianyi/TF-OVCOS
PY=/data/tianyi/conda_envs/tf-ovos/bin/python

echo "[$(date)] waiting for sam_amg_siglip (PID 495453) to finish..."
while kill -0 495453 2>/dev/null; do
  sleep 60
done
echo "[$(date)] sam_amg_siglip finished. Starting serial ADE-847 solo re-runs."

for method in naclip scclip cass trident; do
  echo "[$(date)] starting $method ADE-847 solo re-run..."
  $PY scripts/real_obs5_diagnostic_multi.py \
    --method "$method" \
    --datasets ade847 \
    --num-images 2000 \
    --output-csv "runs/analysis/obs5_real_diagnostic_${method}_ade847_solo.csv" \
    > "runs/logs/obs5_${method}_ade847_solo.log" 2>&1
  echo "[$(date)] finished $method ADE-847 solo re-run (exit=$?)."
done

echo "[$(date)] all ADE-847 solo re-runs complete."
