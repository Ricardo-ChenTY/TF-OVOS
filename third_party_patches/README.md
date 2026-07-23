# Third-party baseline patches

`third_party/official_methods/*` are separate git clones of each baseline's
official repo and are excluded from this repo's git history. This directory
snapshots the local, uncommitted changes made to get each baseline running
under the TF-OVOS benchmark, so they can be restored after a fresh clone of
the official repos.

For each `<Repo>/`:
- `tracked_changes.diff` — modifications to files that exist in the official
  repo. Apply from inside `third_party/official_methods/<Repo>/` with:
  `git apply /path/to/third_party_patches/<Repo>/tracked_changes.diff`
- `untracked/` — new files (mostly `configs/cfg_tfovos_*.py` and class-name
  txt files) that don't exist upstream. Copy the contents into
  `third_party/official_methods/<Repo>/` at matching paths.

Captured 2026-07-23 from the working tree in `runs/analysis` land (Table
2/9 fixes, 6-method Obs 5 diagnostic). Regenerate this snapshot any time
before the local `third_party/official_methods` clones are wiped or moved.
