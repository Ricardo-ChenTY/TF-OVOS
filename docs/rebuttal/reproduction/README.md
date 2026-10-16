# Reproduction handoff (2026-10-16)

This is the source handoff for the future anonymous release. It is **not yet an
anonymous release or a claim that all 17 methods have been installed and rerun on
a new machine**. Frozen artifacts contain historical absolute paths. Preserve
them for provenance; build and validate a separately anonymized export later.

## What is executable now

| Scope | Entry / inputs | What the check establishes |
| --- | --- | --- |
| Expanded E4 table and image-bootstrap intervals | `scripts/reproduce_published_statistics.py`, published result archive | Recomputes 25 latency rows and 10 paired-difference rows from 20,000 timing records; checks all numeric fields at absolute tolerance 1e-12 |
| CLIPtrase corrected class averaging | Same command, published four confusion matrices | Recomputes native and union-supported mIoU, ZIoU and class denominators; does not regenerate predictions |
| Frozen runtime consistency | `scripts/verify_expanded_e4_code.py` | Checks 36 published runtime files against both protocol archives |
| Earlier reviewer evidence | `scripts/verify_reply_support.py` | Checks 48 evidence files, 42 original runtime sources and the original review archive; this is an integrity check, not a full inference replay |
| Four proposal methods + online CorrCLIP, E4 | `scripts/reproduce_e4.py` | Relocates frozen workers, verifies exact assets and inputs, runs one selected job, compares raw prediction-array hashes with published per-image records |
| CorrCLIP source installation | `corrclip_runtime/` | A clean upstream tree plus this patch/overlay reconstructs all 205 native files listed in its frozen E4 source manifest |
| Other native methods, fixed-GT, common-GT, component and thing/stuff audits | `../artifacts/review_20261011_closed.tar.gz`, original report/code and later correction packages | Code/evidence is preserved; their historical launch paths have **not** all received the new portable launcher or a clean-machine installation test |

The five E4 configurations are not a replacement for the historical 18-row
scope-qualified overview. FreeDA also requires its separate environment and
retrieval assets. Do not describe this handoff as a universal one-command runner.

## Validation completed on 2026-10-16

[Machine-readable validation](validation_20261016.json): all five configurations
passed five warmups and one measured VOC image through the new entrypoint. Each
measured output-array hash exactly matches its published record. CorrCLIP used a
separate tree reconstructed from upstream plus the supplied patch. Both protocols
validated all 4,020 selected input image/GT pairs. Eleven targeted tests passed;
the earlier evidence and frozen runtime checks still pass. The CPU replay was
also executed from a copied source tree outside the repository, without `runs/`,
`data/`, `weights/` or `.cache/` directories. This used the existing installed
Python environments; it is not a fresh-machine installation test.

## CPU-only result replay

Use Python with NumPy. No model weights, dataset, historical run folder or GPU is
needed. Run from the repository root; the output directory must not already exist.

```bash
python scripts/verify_reply_support.py
python scripts/verify_expanded_e4_code.py
python scripts/reproduce_published_statistics.py --out /tmp/replayed-statistics
```

The replay checks ordered image IDs across all methods and targets, 1,000 unique
images per target, finite positive times, and the published prediction comparison
reports. It recomputes the 10,000-draw bootstrap with seed 20261015. Those reports
are archived evidence; without raw predictions the CPU command cannot independently
redo the original pixel comparison.

## GPU inference replay of the five E4 configurations

1. Install a compatible GPU environment. `environments/e4_observed.json` records
   the actual Python, CUDA, package versions and extra dependency directory used
   on the server. `environments/freeda_observed.json` is a separate environment.
   These are **inventories, not tested fresh-install lockfiles**. The lightweight
   top-level `environment.yml` alone is not sufficient for all native models.
2. Prepare the licensed validation data using `scripts/dl_*.sh` and
   `scripts/prepare_manifests.py`. The four compact target sizes are 1,449 / 5,105 /
   2,000 / 5,000. The replay uses the exact 1,000 measured IDs and five disjoint
   warmup IDs per target in the published protocol archives; it does not resample.
   Prefix mapping can target each dataset directory independently.
3. Provide the model assets and cache snapshots identified in `model_assets.json`
   and the protocol's `source_sha256.json`. Weights are downloaded separately,
   **not committed to Git**. HF model revisions are recorded in the asset paths;
   do not silently use a newer `main`. The SAM checkpoint is ViT-H for the four
   proposal methods; the online CorrCLIP variant uses SAM2-large, its own CLIP
   checkpoint and DINO ViT-B/8. Trident's ViT-B checkpoint is a different recipe.
4. For online CorrCLIP, check out upstream commit
   `6c1fd2c02ff90f071ed744a5998be0cda7ad75b4` into
   `<official_methods>/CorrCLIP`, apply `corrclip_runtime/tracked_changes.diff`
   from inside that checkout, then copy `corrclip_runtime/untracked/` over it.
   The worker imports its vendored SAM2 code. See the DINO torch-hub source note
   below. The old `third_party_patches/CorrCLIP` is not a substitute for this
   newly verified runtime patch. `official_checkout_inventory.json` records
   the other observed checkout revisions, without claiming their older patches
   reconstruct every current run.
5. Copy `e4_paths.example.json` outside Git and replace its destination paths.
   `path_map` keys are historical prefixes from immutable manifests; values are
   your actual directories. More specific prefixes take precedence. Keep
   `repository`, `official_methods`, `e4_cache` and the asset environment paths
   consistent with those mappings. `extra_python_paths` can be empty if the
   overlay dependencies are installed in the environment. The launcher prepends
   its frozen package and does not import a historical experiment's package.

```bash
python scripts/reproduce_e4.py prepare --group proposal \
  --config /path/to/local-paths.json --run /path/to/new-proposal-run
python scripts/reproduce_e4.py prepare --group corrclip \
  --config /path/to/local-paths.json --run /path/to/new-corrclip-run

# Small initial check: five warmups, then one measured image. This is NOT a full run.
python scripts/reproduce_e4.py run --run /path/to/new-proposal-run \
  --method sam_amg_clip --dataset voc20 --stop-after 1
python scripts/reproduce_e4.py run --run /path/to/new-corrclip-run \
  --method corrclip_online --dataset voc20 --stop-after 1
```

The other proposal method names are `sam_amg_siglip`, `dinov2_sam_clip`, and
`dinov2_sam_siglip`; targets are `voc20`, `context59`, `ade150`, `coco_stuff171`.
Run jobs serially on one otherwise idle GPU. To finish a previously checked job,
rerun its command without `--stop-after`; resume checks reject changed code,
protocols or saved predictions. Removing that flag executes all 1,000 selected
images for that job, **not full-dataset evaluation**. There is no automatic queue.

`prepare` hashes all selected inputs and the archived source/weight inventory.
Byte-identical PNG encoding is required by this strict provenance check: a
semantically equivalent re-encoding still needs an explicitly audited new input
manifest. It does not silently weaken the check. `run` clears inherited model
hyperparameter overrides, restores the recorded per-group TF32 setting, and
uses the original synchronized timer and GPU contention guard. Linux/NVIDIA
NVML is required; multi-GPU logical device remapping has not been validated.

Each job writes `published_prediction_check.json`. A raw-array hash mismatch
(including a dtype difference) fails rather than substituting new quality scores.
Only a complete 1,000-image job provides a full sample comparison. Warmups and
one-image smoke checks do not establish all-target equivalence or fresh-install
reproducibility. Timings need not equal A100 measurements on another machine.
Keep the published full-quality values separate from this timing sample's mIoU.

### Native DINO torch-hub source

CorrCLIP calls `torch.hub.load('facebookresearch/dino:main', ...)`. The server
uses a cached source tree, so leaving this unpinned could load future code.
`dino_hub_source.tar.gz` preserves the observed Python source and upstream license;
extract it into `<TORCH_HOME>/hub/`, producing `facebookresearch_dino_main/`.
This is an additional source inventory captured at handoff, not a retroactive
claim that the old timing manifest hashed this dependency. Its model checkpoint
hash was already in that manifest. Source archive SHA256 is in the adjacent file.

## Dataset and metric contract

- Semantic IDs are zero-based. Compact targets use void 255. Context-459 and
  ADE-847 require uint16 with void 65535; class 255 is a real large-vocabulary
  class. ADE-847 uses `configs/label_maps/ade20k847_official_ids.json`.
- Native CLIPtrase predictions use its 336 center crop; do not stretch them to
  original-frame GT or insert them into the full-image common-GT audit.
- Primary comparable mIoU averages classes with nonzero union; do not mix that
  with the native CLIPtrase GT-supported mean. The CPU replay shows both.
- Fixed-GT evidence uses candidate seed 0; the old all-seed scheduling prototype
  is not the reproduction command for the reported experiment.
- E3 retains declared native presets and disclosed project extensions. This
  handoff does not turn them into a single invariant cross-target recipe.

## Before publishing the reorganized anonymous repository

Keep the original evidence immutable. Export a separate version with neutral
paths, no author-identifying Git history/URLs, and working anonymous evidence
links. Path rewriting changes file hashes: generate a new export manifest and
retain a mapping to the original content hashes. Re-run the CPU checks against
the export and test its actual README commands from a fresh checkout. Prepare
proper environment installation locks and exercise every advertised native
method entry before saying all 17 methods reproduce from scratch. These remaining
release checks do not require launching new scientific comparison experiments.
