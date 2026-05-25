#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/data/tianyi/conda_envs/tf-ovos/bin/python}"
RUNS="$ROOT/runs/official"
LOGS="$ROOT/runs/logs"

mkdir -p "$RUNS" "$LOGS"

echo "[setup] checking mmseg stack"
if ! "$PY" - <<'PYCODE'
import clip  # noqa: F401
import mmcv  # noqa: F401
import mmengine  # noqa: F401
import mmseg  # noqa: F401
PYCODE
then
  echo "[setup] installing official-method common dependencies"
  "$PY" -m pip install -U openmim
  "$PY" -m pip install 'git+https://github.com/openai/CLIP.git'
  "$PY" -m pip install ftfy regex yapf==0.40.1
  "$PY" -m pip install mmengine==0.10.7 mmcv-lite==2.1.0 mmsegmentation==1.2.2
fi

echo "[setup] preparing generated configs"
"$PY" "$ROOT/scripts/prepare_official_mmseg_configs.py"

run_method() {
  local method="$1"
  local repo="$ROOT/third_party/official_methods/$method"
  local method_lower
  method_lower="$(printf '%s' "$method" | tr '[:upper:]' '[:lower:]')"
  local method_log="$LOGS/official_${method_lower}_e1.log"

  {
    echo "[$method] start $(date -Is)"
    cd "$repo"
    for dataset in voc20 context59 ade20k coco_stuff164k; do
      echo "[$method] dataset=$dataset start $(date -Is)"
      if [[ "$method" == "ResCLIP" ]]; then
        "$PY" eval.py \
          --config "configs/cfg_tfovos_${dataset}.py" \
          --arch wo_resi \
          --attn resclip \
          --std 5 \
          --work-dir "$RUNS/$method_lower/$dataset"
      else
        "$PY" eval.py \
          --config "configs/cfg_tfovos_${dataset}.py" \
          --work-dir "$RUNS/$method_lower/$dataset"
      fi
      echo "[$method] dataset=$dataset done $(date -Is)"
    done
    echo "[$method] done $(date -Is)"
  } > "$method_log" 2>&1
}

run_method SCLIP &
run_method NACLIP &
run_method ResCLIP &

wait
echo "[all] official Phase 1 MMSeg jobs finished $(date -Is)"
