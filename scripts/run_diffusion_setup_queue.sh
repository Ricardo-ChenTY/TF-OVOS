#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="$ROOT/runs/logs"
STATUS="$LOGS/diffusion_setup_status.md"
FREE_ROOT="$ROOT/third_party/official_methods/freeda"
FREE_SRC="$FREE_ROOT/src"
OVDIFF_ROOT="$ROOT/third_party/official_methods/ovdiff"
TOOLS_VENV="/data/tianyi/venvs/diffusion_tools"
CONDA="/data/tianyi/miniconda3/bin/conda"

mkdir -p "$LOGS" "$ROOT/cache/diffusion" /data/tianyi/venvs

log() { echo "[diffusion-setup] $* $(date -Is)"; }

wait_for_tmux_done() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    log "waiting for tmux session $session"
    sleep 300
  done
}

link_force() {
  local src="$1"
  local dst="$2"
  if [[ -e "$dst" || -L "$dst" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  ln -s "$src" "$dst"
  log "linked $dst -> $src"
}

write_status_header() {
  cat > "$STATUS" <<EOF
# Diffusion Setup Status

Generated: $(date -Is)

This queue prepares FreeDA/OVDiff after the current proposal queue finishes. It does not run full E1/E2 eval automatically.

EOF
}

append_status() { printf '%s\n' "$*" >> "$STATUS"; }

prepare_data_links() {
  log "preparing dataset symlinks"
  mkdir -p "$FREE_SRC/data" "$OVDIFF_ROOT/data"

  # FreeDA expects paths relative to src/: ./data/VOCdevkit, ./data/ade/ADEChallengeData2016, ./data/coco_stuff164k/{images,annotations}
  link_force "$ROOT/data/raw/VOCdevkit" "$FREE_SRC/data/VOCdevkit"
  mkdir -p "$FREE_SRC/data/ade"
  link_force "$ROOT/data/raw/ADEChallengeData2016" "$FREE_SRC/data/ade/ADEChallengeData2016"
  mkdir -p "$FREE_SRC/data/coco_stuff164k/images"
  link_force "$ROOT/data/raw/coco_stuff171/val2017" "$FREE_SRC/data/coco_stuff164k/images/val2017"
  link_force "$ROOT/data/raw/coco_stuff171/annotations" "$FREE_SRC/data/coco_stuff164k/annotations"

  # OVDiff uses mmcv-style data/ directory.
  link_force "$ROOT/data/raw/VOCdevkit" "$OVDIFF_ROOT/data/VOCdevkit"
}

ensure_tools_venv() {
  if [[ ! -x "$TOOLS_VENV/bin/python" ]]; then
    log "creating lightweight diffusion tools venv"
    /usr/bin/python3 -m venv "$TOOLS_VENV"
  fi
  "$TOOLS_VENV/bin/python" -m pip install -U pip setuptools wheel >/tmp/diffusion_tools_pip.log 2>&1 || return 1
  "$TOOLS_VENV/bin/python" -m pip install -U gdown >/tmp/diffusion_tools_gdown.log 2>&1 || return 1
}

download_freeda_artifacts() {
  local data_dir="$FREE_SRC/data"
  mkdir -p "$data_dir"
  if [[ -f "$data_dir/faiss_index/knn.index" && -d "$data_dir/prototype_embeddings" ]]; then
    log "FreeDA prototype/faiss artifacts already present"
    return 0
  fi

  if ! ensure_tools_venv; then
    log "could not prepare gdown; FreeDA artifacts remain pending"
    return 0
  fi

  log "attempting FreeDA artifact downloads"
  (
    cd "$data_dir"
    if [[ ! -f prototype_embeddings.tar && ! -d prototype_embeddings ]]; then
      "$TOOLS_VENV/bin/gdown" --fuzzy 'https://drive.google.com/file/d/1U4d0exJuq29b0rLR6iOT20ErW3DAmgw0/view?usp=sharing' -O prototype_embeddings.tar || exit 0
    fi
    if [[ -f prototype_embeddings.tar && ! -d prototype_embeddings ]]; then
      mkdir -p prototype_embeddings
      tar -xzf prototype_embeddings.tar -C prototype_embeddings || true
    fi
    if [[ ! -f faiss_index.zip && ! -d faiss_index ]]; then
      "$TOOLS_VENV/bin/gdown" --fuzzy 'https://drive.google.com/file/d/1FHjpM0aqPf9OjiuG_341EMlEuq6hsh6L/view?usp=sharing' -O faiss_index.zip || exit 0
    fi
    if [[ -f faiss_index.zip && ! -d faiss_index ]]; then
      unzip -q faiss_index.zip -d faiss_index || true
    fi
  ) || true
}

download_ovdiff_cutler() {
  local ckpt="$OVDIFF_ROOT/CutLER/cutler_cascade_final.pth"
  if [[ -f "$ckpt" ]]; then
    log "OVDiff CutLER checkpoint already present"
    return 0
  fi
  log "attempting OVDiff CutLER checkpoint download"
  wget -c -P "$OVDIFF_ROOT/CutLER" http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth || true
}

summarize_state() {
  write_status_header
  append_status "## Repositories"
  append_status "- FreeDA: $FREE_ROOT"
  append_status "- OVDiff: $OVDIFF_ROOT"
  append_status ""
  append_status "## Data Links"
  for p in \
    "$FREE_SRC/data/VOCdevkit" \
    "$FREE_SRC/data/ade/ADEChallengeData2016" \
    "$FREE_SRC/data/coco_stuff164k/images/val2017" \
    "$FREE_SRC/data/coco_stuff164k/annotations" \
    "$OVDIFF_ROOT/data/VOCdevkit"; do
    if [[ -e "$p" || -L "$p" ]]; then append_status "- OK: $p"; else append_status "- MISSING: $p"; fi
  done
  append_status ""
  append_status "## Required Artifacts"
  if [[ -f "$FREE_SRC/data/faiss_index/knn.index" ]]; then append_status "- OK: FreeDA faiss index"; else append_status "- PENDING: FreeDA faiss index"; fi
  if [[ -d "$FREE_SRC/data/prototype_embeddings" ]]; then append_status "- OK: FreeDA prototype embeddings"; else append_status "- PENDING: FreeDA prototype embeddings"; fi
  if [[ -f "$OVDIFF_ROOT/CutLER/cutler_cascade_final.pth" ]]; then append_status "- OK: OVDiff CutLER checkpoint"; else append_status "- PENDING: OVDiff CutLER checkpoint"; fi
  append_status ""
  append_status "## Environment Notes"
  append_status "- FreeDA wants PyTorch 1.13.1, mmcv-full 1.6.2, mmsegmentation 0.27.0."
  append_status "- OVDiff wants PyTorch 1.12.1, diffusers 0.14.0, mmcv-full 1.7.1, mmsegmentation 0.30.0, detectron2 0.6."
  append_status "- Do not use the current tf-ovos env for these old stacks. Use isolated envs under /data/tianyi/conda_envs."
  append_status ""
  append_status "## Recommended Next Smoke Commands"
  append_status '```bash'
  append_status '# FreeDA VOC smoke/full eval entry after env is built:'
  append_status 'cd /data/tianyi/TF-OVCOS/third_party/official_methods/freeda/src'
  append_status 'python -m torch.distributed.run main.py --eval --eval_cfg configs/pascal20/freeda_pascal20.yml --eval_base_cfg configs/pascal20/eval_pascal20.yml'
  append_status ''
  append_status '# OVDiff VOC staged entry after env is built:'
  append_status 'cd /data/tianyi/TF-OVCOS/third_party/official_methods/ovdiff'
  append_status 'python sample_support_set.py voc outputs/voc'
  append_status '```'
}

main() {
  log "diffusion setup queue started"
  wait_for_tmux_done proposal_e1_queue
  prepare_data_links
  download_freeda_artifacts
  download_ovdiff_cutler
  summarize_state
  df -h /data >> "$STATUS" || true
  du -sh "$FREE_ROOT" "$OVDIFF_ROOT" >> "$STATUS" || true
  log "diffusion setup queue finished; see $STATUS"
}

main "$@"
