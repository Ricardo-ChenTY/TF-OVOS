#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/tianyi/TF-OVCOS
PY=/data/tianyi/conda_envs/tf-ovos/bin/python
FREE_PY=/data/tianyi/conda_envs/freeda-official-py38-t112b/bin/python
ART="$ROOT/runs/artifacts/official_predictions"
LOGS="$ROOT/runs/logs"
RUNS="$ROOT/runs/official"
mkdir -p "$LOGS" "$ART"
LOG="$LOGS/missing_mcmr_artifacts_queue.log"

log() { echo "[missing-mcmr] $* $(date -Is)" | tee -a "$LOG"; }

wait_session_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session $session"
    sleep 300
  done
}

artifact_done() {
  local method="$1" dataset="$2" total="$3"
  local dir="$ART/$method/$dataset"
  [[ -d "$dir" ]] || return 1
  local n
  n=$(find "$dir" -maxdepth 1 -type f -name '*.png' | wc -l)
  [[ "$n" -ge "$total" ]]
}

run_mmseg_method_dataset() {
  local method="$1" repo="$2" dataset="$3" total="$4"
  local config="configs/cfg_tfovos_${dataset}.py"
  local work_dir="$RUNS/$method/$dataset"
  local log_file="$LOGS/mcmr_artifact_${method}_${dataset}.log"
  if artifact_done "$method" "$dataset" "$total"; then
    log "skip $method $dataset; artifacts complete"
    return
  fi
  log "start $method $dataset artifact export"
  (
    cd "$repo"
    if [[ "$method" == "cass" ]]; then
      "$PY" eval.py --config "$config" --pamr off --work-dir "$work_dir"
    else
      "$PY" eval.py --config "$config" --work-dir "$work_dir"
    fi
  ) > "$log_file" 2>&1 || log "FAILED $method $dataset; see $log_file"
  log "done $method $dataset artifact export"
}

run_cliptrase_dataset() {
  local dataset="$1" out_dataset="$2" total="$3"
  local repo="$ROOT/third_party/official_methods/CLIPtrase"
  local log_file="$LOGS/mcmr_artifact_cliptrase_${out_dataset}.log"
  if artifact_done cliptrase "$out_dataset" "$total"; then
    log "skip cliptrase $out_dataset; artifacts complete"
    return
  fi
  log "start cliptrase $dataset artifact export"
  (
    cd "$repo"
    CLIPTRASE_SAVE_ROOT="$ART" "$PY" -c "
import sys, torch
sys.path.insert(0, '.')
import clip_utils
from clip_self_correlation import self_clip
clip_model, _ = clip_utils.load('ViT-B/16', image_size=224)
clip_model = clip_model.to('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    self_clip(clip_model, '$dataset', image_size=336, eps=1.1, min=7)
"
  ) > "$log_file" 2>&1 || log "FAILED cliptrase $dataset; see $log_file"
  log "done cliptrase $dataset artifact export"
}

run_freeda_eval() {
  local name="$1" dataset="$2" eval_cfg="$3" eval_base_cfg="$4" total="$5"
  local src="$ROOT/third_party/official_methods/freeda/src"
  local out="$ROOT/runs/freeda_artifact/$name"
  local log_file="$LOGS/mcmr_artifact_freeda_${dataset}.log"
  if artifact_done freeda "$dataset" "$total"; then
    log "skip freeda $dataset; artifacts complete"
    return
  fi
  log "start freeda $dataset artifact export"
  (
    cd "$src"
    FREEDA_SAVE_ROOT="$ART" FREEDA_DATASET_NAME="$dataset" "$FREE_PY" -m torch.distributed.run --nproc_per_node=1 main.py \
      --eval --eval_cfg "$eval_cfg" --eval_base_cfg "$eval_base_cfg" --output "$out"
  ) > "$log_file" 2>&1 || log "FAILED freeda $dataset; see $log_file"
  log "done freeda $dataset artifact export"
}

main() {
  log "queue start"
  wait_session_done table10_hard_domain_appendix

  "$PY" "$ROOT/scripts/prepare_trident_cass_configs.py"
  "$PY" "$ROOT/scripts/prepare_scclip_configs.py"

  for dataset in voc20 context59 ade20k coco_stuff164k context459 ade847; do
    case "$dataset" in
      voc20) total=1449 ;;
      context59) total=5105 ;;
      ade20k) total=2000 ;;
      coco_stuff164k) total=5000 ;;
      context459) total=5105 ;;
      ade847) total=2000 ;;
    esac
    run_mmseg_method_dataset trident "$ROOT/third_party/official_methods/Trident" "$dataset" "$total"
    run_mmseg_method_dataset scclip "$ROOT/third_party/official_methods/SC-CLIP" "$dataset" "$total"
    run_mmseg_method_dataset cass "$ROOT/third_party/official_methods/CASS" "$dataset" "$total"
  done

  run_cliptrase_dataset VOC20 voc20 1449
  run_cliptrase_dataset PC59 context59 5105
  run_cliptrase_dataset ADE150 ade20k 2000
  run_cliptrase_dataset COCO171_val coco_stuff164k 5000
  run_cliptrase_dataset PC459 context459 5105
  run_cliptrase_dataset ADEfull ade847 2000

  GEN="$ROOT/runs/freeda/generated_configs"
  run_freeda_eval voc20 voc20 configs/pascal20/freeda_pascal20.yml configs/pascal20/eval_pascal20.yml 1449
  run_freeda_eval context59 context59 configs/pascal59/freeda_pascal59.yml "$GEN/eval_context59.yml" 5105
  run_freeda_eval ade20k150 ade20k configs/ade/freeda_ade.yml configs/ade/eval_ade.yml 2000
  run_freeda_eval coco_stuff171 coco_stuff164k configs/cocostuff/freeda_cocostuff.yml configs/cocostuff/eval_cocostuff.yml 5000
  run_freeda_eval context459 context459 configs/pascal59/freeda_pascal59.yml "$GEN/eval_context459.yml" 5105
  run_freeda_eval ade847 ade847 configs/ade/freeda_ade.yml "$GEN/eval_ade847.yml" 2000

  log "run diagnostic tables"
  "$PY" "$ROOT/scripts/generate_diagnostic_tables.py" --methods sclip naclip resclip proxyclip corrclip scclip cliptrase trident cass freeda --datasets voc20 context59 ade20k coco_stuff164k context459 ade847 > "$LOGS/mcmr_artifact_generate_diagnostics.log" 2>&1 || log "diagnostic generation failed"
  "$PY" "$ROOT/scripts/analyze_official_results.py" > "$LOGS/mcmr_artifact_analyze.log" 2>&1 || true
  log "queue done"
}

main "$@"
