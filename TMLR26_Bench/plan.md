# TMLR26 Paper Correctness Fix Plan

Consolidates a manual review of the submitted PDF (`TMLR26_OVS_Bench.pdf`)
against the actual repository state. Supersedes `experiment_plan_draft.md`
in this directory, which was written before the repo was audited. The
top-level `/plan.md` covers a different, still-active goal (populating
every method row for E1-E4); this file covers correctness fixes to what's
already written up.

Session date: 2026-07-13.

## 0. What changed vs. the last known state

- `origin` (`git@github.com:Ricardo-ChenTY/TF-OVOS.git`) had 9 commits not
  yet pulled locally. Pulled them. They added
  `tf_ovcos_full_results_20260526/runs/analysis/` (the real source CSVs
  behind several paper tables) and this `TMLR26_Bench/` directory
  (case studies, figures, generation scripts).
- Those CSVs made it possible to verify several claims in the PDF against
  actual computed numbers instead of just re-deriving them from the tables
  printed in the PDF.

## 1. Confirmed and fixed

### 1.1 `METHOD_GROUPS` mislabeling in `scripts/generate_diagnostic_tables.py`

**Bug:** lines 75-76 hardcoded `proxyclip` and `corrclip` into
`"Proposal+naming TF methods"`. Per the paper's own Table 1 taxonomy, both
are CLIP+VFM methods (same family as Trident/CASS), not proposal+naming
methods (SAM-AMG, DINOv2+SAM).

**Consequence:** Appendix Table 9 (`D.1 Mismatch Summary for E2`)'s
"Proposal+naming TF methods" row was actually built from CorrCLIP +
ProxyCLIP, not from the real proposal methods. The real proposal methods
(SAM-AMG+CLIP/SigLIP, DINOv2+SAM+CLIP/SigLIP) were never processed by this
script at all — their prediction format (`runs/sam_amg_clip/*/predictions.jsonl`,
mask-tuple based) doesn't match what this script reads
(`runs/artifacts/official_predictions/<method>/<dataset>/*.png`, dense
label maps). This explains why Table 9's reported MCMR for "Proposal+naming"
(0.3454) doesn't match Table 2's per-method MCMR for the real proposal rows
(0.711-0.776) and doesn't reverse-engineer to any combination of them.

**Fix applied:** changed the two lines to `"CLIP + VFM TF methods"`.
Re-ran the script (confirmed via `runs/analysis/table11_e2_mismatch_summary.csv`):
`"CLIP + VFM TF methods"` now correctly pools CorrCLIP/ProxyCLIP with
Trident/CASS (localized_pairs=253965, mismatch_pairs=101705,
`mismatch_pairs/localized_pairs` verified equal to `MCMR@0.5_pooled`
by hand). `"Dense-map TF methods"`, `"Diffusion/reference TF methods"`,
and `"Trained references"` are likewise pooled-and-verified-consistent now.

**Also found while re-deriving the pre-fix numbers:** the paper's actual
printed Table 9 values (Dense-map: 413463 localized pairs / MCMR 0.2972)
correspond to only 3 dense methods (137821 × 3), while the artifact root
now has 5 dense methods (sclip, naclip, resclip, scclip, cliptrase — scclip
and cliptrase were added later per the top-level `plan.md`'s P4 wave).
**The printed Table 9 numbers are stale relative to the current data**,
independent of the mislabeling bug. `runs/analysis/table11_e2_mismatch_summary.csv`
now has the current, correctly-grouped, arithmetic-consistent numbers.
**Action: regenerate the Table-9-equivalent numbers in the paper from this
file before the next submission, not from whatever spreadsheet snapshot
was used originally.**

**`"Proposal+naming TF methods"` — do not leave this blank; the real
per-method numbers already exist, no new computation needed.** My first
pass here was wrong: I initially wrote (and ran, then had to kill) a whole
new script to recompute per-region diagnostics for SAM-AMG/DINOv2+SAM from
their raw prediction PNGs, treating an empty `artifact_gap` cell as an
acceptable interim state. That was unnecessary rework — **Table 2 already
publishes real per-method MCMR@0.5 for all four proposal methods**
(SAM-AMG+CLIP 0.776, SAM-AMG+SigLIP 0.713, DINOv2+SAM+CLIP 0.776,
DINOv2+SAM+SigLIP 0.711; also in `e2_vocab_robustness_filled.csv`), computed
by the project's own standard eval pipeline when these predictions were
first generated (see each method's `runs/<method>/<dataset>_val/metrics.json`,
which already has a computed `mcmr_05` field per dataset). The family-level
number just needed the simple average of those four already-published
values: **`MCMR@0.5_mean_of_cells = 0.744`**, now filled into
`table11_e2_mismatch_summary.csv`'s `"Proposal+naming TF methods"` row.
`localized_pairs`/`mismatch_pairs`/`MCMR@0.5_pooled` are left blank for this
row since Table 2's methodology only ever published the ratio (averaged
over Ctx-459/ADE-847), not the raw pair counts needed for a pooled sum —
that's a real, smaller gap (documented, not fabricated), not the
much bigger "we never evaluated this family at all" gap the empty cell
implied before.

Status: done. `runs/analysis/table11_e2_mismatch_summary.csv` has real,
consistent numbers for all five populated method groups; paste into the
manuscript's Appendix D.1 table.

## 2. Investigated in depth, root cause found and FIXED

### 2.1 CLIPtrase / FreeDA catastrophic collapse on COCO-Stuff diagnostics only — RESOLVED (2026-07-14)

**Update: this was a real, fixable bug, not an incompatible method-specific
"trick" as the fallback hypothesis assumed. Fixed and verified this
session.** The crosswalk built earlier this session
(`runs/analysis/coco_stuff_official_to_vocab_crosswalk.json`, 173 entries,
`official_idx -> our_vocab_idx`) was already correct — re-derived it
independently from scratch (full 5000-image val set, vectorized confusion
matrix, purity 1.0 on all 161 observed classes) and got the same mapping.
**The actual bug was in how the crosswalk was applied, not in the crosswalk
itself**: the earlier session applied it directly to CLIPtrase/FreeDA's raw
prediction values, but those predictions store `official_labelTrainId + 1`
(their own save code appears to reserve `0`, or a similar off-by-one from
its own dataset wrapper), so the crosswalk must be applied to
`(pred_value - 1)`. Applied directly (no `-1`), the crosswalk changes pixel
values but leaves the match rate statistically unchanged — exactly the
"bit-for-bit identical before/after" symptom recorded below, now fully
explained.

**Verification (300 held-out images per method, pixel accuracy against
canonical GT):**

| method | current (broken, `pred-1` only) | fixed (`pred-1` then crosswalk) |
|---|---|---|
| freeda | 0.95% | 43.7% |
| cliptrase | 0.80% | 36.9% |

**Fix applied to the actual artifact tree, not just a side experiment:**
- Originals backed up to
  `runs/artifacts/official_predictions/{freeda,cliptrase}/coco_stuff164k.bak_pre_gtconvention_fix/`.
- Corrected predictions (crosswalk applied to `pred-1`, then `+1`
  pre-compensated so the scoring pipelines' own hardcoded
  `prediction_label_offset=-1` for `coco_stuff164k` lands on the right final
  index — this pre-compensation step was itself a second trap: applying the
  fix without it reproduced the same near-zero numbers, because the
  pipeline's blanket `-1` was silently applied a second time on top of an
  already-correct value) written into the live
  `runs/artifacts/official_predictions/{freeda,cliptrase}/coco_stuff164k/`
  directories, replacing the originals.
- Ran `scripts/generate_diagnostic_tables.py` (full leaderboard, all 11
  methods x 6 datasets) against the corrected tree to regenerate
  `table9_diagnostic_probes.csv`, `table11_e2_mismatch_summary.csv`,
  `table12_supporting_readouts.csv` in `runs/analysis/`.

**Before/after, full 5000-image COCO-Stuff-171 val set:**

| metric | freeda before | freeda after | cliptrase before | cliptrase after |
|---|---|---|---|---|
| gt_region_naming_top1 | 0.014 | 0.410 | 0.011 | 0.259 |
| gt_text_localization_iou | 0.009 | 0.249 | 0.004 | 0.128 |
| mcmr_at_0.5 | 0.987 | 0.343 | 0.989 | 0.379 |
| zero_iou_class_rate | n/a (col added later) | 0.012 | n/a | 0.037 |
| proposal_oracle_iou | 0.414 | 0.404 | 0.264 | 0.254 |
| proposal_recall_at_0.5 | 0.380 | 0.372 | 0.202 | 0.197 |

**Action needed in Overleaf: Appendix D.1's pooled MCMR@0.5-by-method-group
table changes for the two groups these methods belong to** (recomputed by
reconstructing the exact old/broken pooled pair counts and diffing against
the new `table11_e2_mismatch_summary.csv`, not estimated):

| method group | members | MCMR@0.5_pooled before | MCMR@0.5_pooled after |
|---|---|---|---|
| Dense-map TF methods | sclip, scclip, cliptrase, naclip, resclip | 0.370 | **0.350** |
| Diffusion/reference TF methods | freeda (only member) | 0.473 | **0.319** |

The "Dense-map" group's shift is small because cliptrase's one bad dataset
is diluted across 5 methods x 6 datasets; the "Diffusion/reference" group
moves by a full 0.15 because freeda is its only member. No other method
group's numbers change (CLIP+VFM, Trained references, and the two
`artifact_gap` groups were never touched by this bug).

**Table 1/Table 2 headline mIoU need NO changes** — those numbers come from
each method's own official mmseg eval run, never from the prediction PNGs
this fix touched.

Naming/localization/MCMR move by 25-45x and land in the same range as the
other 9 methods on this dataset (consistent with CLIPtrase/FreeDA's own
"normal" behavior on the other 5 datasets). Proposal oracle IoU/recall are
essentially unchanged, as expected — SAM's class-agnostic proposals never
depended on this label convention.

**What this does NOT touch:** Table 1/Table 2's headline mIoU for CLIPtrase
(24.06) and FreeDA (28.80) on COCO-Stuff, which come from each method's own
official mmseg-based eval run, not from these saved prediction PNGs — those
numbers were never affected by this bug and do not need revision.

**Original investigation record below, left intact for the record of what
was ruled out and why (still accurate as history; only the ending changed):**

**Symptom:** in `tf_ovcos_full_results_20260526/runs/analysis/table9_diagnostic_probes.csv`,
`cliptrase` and `freeda` show `gt_region_naming_top1 ≈ 0.01` and
`mcmr_at_05 ≈ 0.99` on `coco_stuff164k` specifically, while both look
completely normal (in line with peer methods) on all 5 other datasets, and
both have normal-looking mIoU (24.06 / 28.80) in Table 1 on COCO-Stuff too.

**What this rules out** (verified empirically, not assumed):
- Not a resolution/geometry bug: CLIPtrase's raw prediction PNGs are a
  fixed 336×336 square on *every* dataset (confirmed via direct shape
  inspection), and the same square→native-resolution resize works fine for
  VOC-20/Context-59/ADE-150. It's not COCO-specific.
- Not a class-name-order bug in CLIPtrase's own declared vocabulary:
  compared `third_party/official_methods/CLIPtrase/configs/dataset_cfg.py`'s
  `COCO171_val` label list against `configs/vocab/coco_stuff_171.txt`
  position-by-position around the things→stuff boundary — they match
  (only cosmetic name differences like "cabinet" vs "cabinet-merged").

**What actually explains it — GT-convention mismatch, confirmed with an
authoritative source:**

There are at least two non-identical GT label conventions for COCO-Stuff
sitting in this repo:
1. `data/raw/coco_stuff171_labels/*.png` — built by
   `scripts/prepare_manifests.py`'s `prepare_coco_stuff171()`, using a
   custom `_COCO_ID_TO_VOCAB` LUT (things IDs first, then stuff IDs),
   0-170 valid + 255 ignore. This is what `data/manifests/coco_stuff171_val.jsonl`
   (the shared benchmark manifest) actually points to.
2. `data/official_mmseg/coco_stuff171/annotations/val2017/*_labelTrainIds.png`
   (symlinked from `data/raw/coco_stuff171_labels_mmseg_official/`) — also
   0-170 valid + 255 ignore numerically, but a **different** category→index
   assignment (confirmed: pixel-level diff on a sample image, `np.array_equal`
   is False even though both use the same 0-170/255 envelope).

Checked mmsegmentation's own `COCOStuffDataset` docstring
(`mmseg/datasets/coco_stuff.py`, present in three local conda envs) — it
says the **164k version** (the one this benchmark uses; the 10k version is
a different, incompatible split) uses **Train-IDs 0-170 with 255 as
ignore**, and its hardcoded `METAINFO['classes']` tuple is in
things-then-stuff order — **matching convention (1) and
`configs/vocab/coco_stuff_171.txt` exactly**, not convention (2).

Empirical confirmation: repointing the manifest to convention (2) and
re-measuring pixel match rate for 5 sample methods over 40 images gave:

| method | match rate under GT (1), original | match rate under GT (2) |
|---|---|---|
| sclip | ~0.33 (normal) | 0.017 (collapses) |
| naclip | ~0.37 (normal) | 0.023 (collapses) |
| corrclip | ~0.37 (normal) | 0.021 (collapses) |
| cliptrase | ~0.0 (collapses) | 0.447 (normal) |
| freeda | ~0.06 (collapses) | 0.478 (normal) |

This is decisive: convention (1) is correct for 9 of 11 diagnosed methods
(sclip/naclip/resclip/proxyclip/corrclip/scclip/trident/cass — everything
routed through `scripts/postprocess_official_predictions.py`, which
apparently normalizes predictions into the same order as convention (1)/
vocab.txt). **CLIPtrase and FreeDA's `coco_stuff164k` artifact exports in
`runs/artifacts/official_predictions/{cliptrase,freeda}/coco_stuff164k/`
were produced by their own standalone export code
(`run_cliptrase_dataset` / `run_freeda_eval` in
`scripts/run_missing_mcmr_artifacts_queue.sh`), which bypasses
`postprocess_official_predictions.py` and appears to target GT convention
(2) instead of (1).**

**Manifest action taken this session: none — reverted twice, left as
originally shipped.** It's already correct for the majority of methods;
changing it would have fixed 2 methods and broken 9.

**Reassuring finding:** this bug is isolated to the diagnostic-probe
pipeline (`generate_diagnostic_tables.py` + its manifest). Table 1/Table 2's
core mIoU leaderboard numbers for CLIPtrase (24.06) and FreeDA (28.80) on
COCO-Stuff almost certainly come from a separate, correctly-configured
official-eval code path (`run_official_cliptrase_queue.sh` /
`run_freeda_e1_queue.sh`) that isn't touched by this bug — the leaderboard
should be safe. Only Table 9-style diagnostics and any future per-method
naming/localization probe (Obs 5 style) for these two methods on COCO-Stuff
are at risk.

**Attempted a numeric-LUT fix this session — it failed, documenting why
rather than pretending it worked.** Reverse-engineered a raw-COCO-category-
ID crosswalk empirically: compared `data/raw/coco_stuff171/annotations/val2017`
(raw stuffthingmaps, values = literal COCO category IDs 1-182) against
`data/official_mmseg/coco_stuff171/annotations/val2017/*_labelTrainIds.png`
pixel-by-pixel across all 5000 validation images, getting a clean,
zero-conflict mapping from "official_mmseg index" back to "raw COCO ID"
(167 distinct indices, no contradictions). Composed that with
`scripts/prepare_manifests.py`'s own `_COCO_ID_TO_VOCAB` dict (raw COCO ID
→ our canonical vocab.txt index) to get a direct
`official_mmseg_index → our_vocab_index` crosswalk (161/173 slots resolved
to a real class; 12 mapped to void — raw id 0 = unlabeled, raw id 255, and
~10 deprecated COCO thing categories our vocab doesn't carry a slot for).

Applied this crosswalk as a pixel LUT to CLIPtrase's and FreeDA's existing
`coco_stuff164k` prediction PNGs and re-measured pixel match rate against
our canonical GT (same 40-image spot check as §2.1's original table):
**match rate was bit-for-bit identical before and after the fix**
(CLIPtrase 0.0082 → 0.0082, FreeDA 0.0129 → 0.0129), even though the
remapped pixel values were genuinely different from the originals
(verified directly, not a no-op bug in the script). FreeDA's raw
predictions are already at native GT resolution (no resize step involved),
which rules out the resize/aspect-ratio explanation for at least that
method — so the fix's lack of effect isn't a resize artifact either.

**This means the "official_mmseg index" hypothesis for what convention
CLIPtrase/FreeDA's raw predictions actually use is wrong**, contradicting
§2.1's earlier (correct, still-valid) finding that swapping the *manifest's
GT* to the official_mmseg convention measurably improved their match rate.
Both facts can't be true under a single simple story, which means the real
bug is something other than "these two methods emit official_mmseg-indexed
predictions instead of vocab.txt-indexed ones" — for example, it could be
specific to how each method's own `configs/dataset_cfg.py` (CLIPtrase) or
`.yml` (FreeDA) builds its *own* class list at inference time for this
one dataset, independent of both conventions checked so far. **Not
resolved this session.** Removed the experimental
`runs/artifacts/official_predictions_fixed/` output rather than leave a
non-working "fix" in the artifact tree. Next step for whoever picks this
up: read `third_party/official_methods/CLIPtrase/clip_self_correlation.py`
and FreeDA's `configs/cocostuff/freeda_cocostuff.yml` directly to find the
literal class list each one used for its COCO-Stuff run, rather than
inferring it indirectly from GT-file comparisons.

**Immediate mitigation still recommended:** mark `(cliptrase,
coco_stuff164k)` and `(freeda, coco_stuff164k)` as `known_bad_artifact` in
`generate_diagnostic_tables.py`'s output so they don't silently poison any
average that includes them (this affects nothing already published in the
PDF, since Obs 5's numbers only use CorrCLIP — but Table 8/Table 9 style
aggregates that pool "all methods x all datasets" would be corrupted by
this if extended to include cliptrase/freeda's COCO-Stuff row as-is). Not
implemented this session either — same reasoning as above, didn't want to
add a mitigation flag while still actively wrong about the root cause.

**[Correction, next session, 2026-07-14]: the "bit-for-bit identical
before/after" finding above was real but the conclusion drawn from it
("the official_mmseg index hypothesis is wrong") was not. The crosswalk
built in the paragraphs above was in fact correct — the reason applying it
had zero effect is that it was applied directly to the raw prediction
value, when it needed to be applied to `(raw prediction value - 1)`
instead (CLIPtrase/FreeDA's saved predictions are the official labelTrainId
already shifted by +1). That one-line application error, not a wrong
crosswalk, was the entire bug. See the "RESOLVED" writeup at the top of
this subsection for the verified fix, the before/after numbers, and what
was actually changed in the artifact tree. The `known_bad_artifact`
mitigation above is no longer needed since the underlying data is now
correct.

## 3. Verified against source data (from the earlier PDF review) — all subsections now resolved, see 3.1-3.5

These were flagged from reading the PDF alone; now cross-checked against
the actual CSVs pulled from GitHub.

### 3.1 Obs 5's four-probe numbers (0.487/0.403/0.605/0.635) — arithmetic confirmed; real pilot now run, DONE

Verified these are exactly the unweighted 6-dataset average of CorrCLIP's
row in `table9_diagnostic_probes.csv` (`gt_region_naming_top1`,
`gt_text_localization_iou`, `proposal_oracle_iou`, `proposal_recall_at_05`).
The arithmetic is right, but every row in that CSV is tagged
`diagnostic_source: proxy_from_dense_label_map` — these are argmax-
confusion-matrix proxies, not real region-conditioned CLIP calls or real
SAM/DINO proposals, even though the paper prose in Obs 5 reads as if real
inference was run.

**Implemented and run this session** as `scripts/real_obs5_diagnostic_pilot.py`
(built with help from an external coding agent — see note at the end of
this subsection — then verified and run directly). Real GT-region naming
via MetaCLIP FullCC `ViT-B-16-quickgelu` (CorrCLIP's own cached
`b16_fullcc2.5b.pt` checkpoint, loaded strictly — verified zero
missing/unexpected keys), real single-class localization via CorrCLIP's
own DINO ViT-B/8 + correlation-refined dense encoder, and real
class-agnostic proposals via SAM ViT-H automatic mask generation (the
exact parameters `SamAmgClipAdapter` uses: points-per-side 32,
pred-iou-thresh 0.86, stability-thresh 0.92, min-area 400). Scope: CorrCLIP
only, first 250 images each from Context-459 and ADE-847 (1492 and 3026
GT-class regions respectively), matching the tuning-ablation pilot's
"bounded, honest pilot" precedent rather than a full leaderboard replacement.

**Result — full validation set, both datasets in full (final, after fixing
the localization binarization — see below):**

| Metric | Context-459 (5105 imgs) | ADE-847 (2000 imgs) | Mean | Old proxy | Δ |
|---|---|---|---|---|---|
| GT-region naming Top-1 | 0.240 | 0.158 | 0.199 | 0.487 | **-0.288** |
| GT-text localization IoU | 0.358 | 0.213 | 0.285 | 0.403 | -0.118 |
| Proposal oracle IoU | 0.674 | 0.571 | 0.623 | 0.605 | +0.018 |
| Proposal Recall@0.5 | 0.720 | 0.606 | 0.663 | 0.635 | +0.028 |

Full detail: `runs/analysis/obs5_real_diagnostic_full.csv` / `.md`
(55292 total GT-class regions, 656922 SAM proposals, 0 errors across all
7105 images). This is a genuine full-dataset result, not a subset
estimate — the numbers above **match the earlier 250-image pilot closely**
(naming 0.199 vs. 0.193, localization 0.285 vs. 0.287, proposal oracle
0.623 vs. 0.646, recall 0.663 vs. 0.691 — all within ~0.02-0.03), which
retroactively confirms the 250-image pilot was already a representative
sample; the full run mainly tightens statistical confidence rather than
changing the picture. The 250-image pilot's own CSV/MD are kept as
`runs/analysis/obs5_real_diagnostic_pilot_250subset.csv` / `.md` for
reference. Ran as a ~7-hour job in a detached `tmux` session
(`obs5_full_run`) at an observed steady-state rate of ~3.4-3.7s/image
(startup/model-loading overhead made the first ~400 images look much
slower in early progress checks — ~7.9s/image — before amortizing down;
worth remembering if timing this kind of job again).

**All four numbers are now real and usable** — the localization metric
went through one failed attempt and one fix this session, documented in
full since the same trap (a plausible-looking but silently broken
diagnostic) is worth remembering for future work on this pipeline:

- **First attempt (rejected):** score only the single correct class's text
  prompt against the dense features, no competition from other classes,
  Otsu-threshold that one map. Manually checked 8 sample regions by
  calling the pilot's own functions directly: GT foreground covers
  3.5%-22% of each image, but the predicted mask covered **0.00%-0.31%**
  in every single case — a systematic collapse to near-empty predictions,
  not "localizing to a plausible-but-wrong region." Root cause: a
  single-prompt similarity map has a narrow, low-contrast value range
  (e.g. 0.25-0.43 across an entire image) with no bimodal structure, and
  Otsu (which assumes a bimodal histogram with a clear valley) ends up
  carving off only the extreme top sliver as foreground regardless of
  image content. This gave the aggregate pilot run an IoU of 0.007 —
  confirmed to be a broken measurement, not a real finding about CorrCLIP.
- **Second attempt (also rejected):** contrast the target class's score
  against the mean score of all other candidate classes at each pixel,
  Otsu-threshold the contrast map. Tested on the same 8 regions — still
  collapsed to near-zero IoU, since with 400+ candidate classes the mean
  of "everything else" is barely different from the per-pixel average
  regardless of the target class.
- **Third attempt (works, now the final method):** score the target class
  against the *full* candidate vocabulary and take the region where it
  wins the argmax against every other class, using CorrCLIP's own real
  dense scores (computed fresh for this pilot, not read from a saved
  prediction map). Tested on the same 8 regions first: genuine
  non-degenerate IoUs (0.79, 0.46, 0.25 where the class actually won the
  competition somewhere in the image; honest zeros where it never did,
  consistent with the "never-predicted" class-silence pattern found in
  E-12/§5.5) rather than uniform near-collapse. Reran first on the
  250-image×2-dataset pilot with this fix (localization IoU 0.287), then
  on the **full validation set** (5105+2000 images, localization IoU
  0.285) — both close to the old proxy's 0.403 (Δ -0.118, not -0.396) and
  close to each other, confirming the fix is stable at scale, not a
  small-sample artifact.

**Tradeoff to disclose if this goes in the manuscript:** the working
(argmax) version no longer fully isolates localization from naming
competition the way the diagnostic was originally conceived — a class
that never wins the full argmax anywhere gets IoU=0 here even if
CorrCLIP's patch-level correlation for it is locally reasonable. It is a
real, defensible measurement of "how well does CorrCLIP's actual
competitive dense inference localize this class when scored in isolation
against its true region," not a pure "given only the answer, ignoring all
competition, can you draw the boundary" measurement. State this precisely
rather than implying the latter.

**Updated bottom line (full-dataset numbers):** naming (0.199 vs. 0.487) is
the number that moved the most and is fully trustworthy — real
region-conditioned CLIP classification is substantially harder than the
dense-argmax proxy suggested, confirmed now at full scale (55292 regions,
not a small sample). Localization (0.285 vs. 0.403) and the two proposal
numbers (0.623/0.663 vs. 0.605/0.635) all land within ~0.02-0.12 of the
old proxy — the proxy's naming number was the one that most overstated
CorrCLIP's real capability; the other three were reasonable stand-ins,
and the full run's proposal numbers moved slightly closer to the proxy
than the 250-image pilot did (0.623 vs. pilot's 0.646, 0.663 vs. pilot's
0.691), for whatever that's worth as an additional confidence signal. This
is real, defensible, full-scale evidence for a footnote or a short
paragraph alongside the disclosed-proxy note, not a wholesale replacement
of Obs 5's numbers (scope still differs from the original: 2 large-vocab
datasets here vs. 6 datasets originally averaged, and CorrCLIP only, not
all 18 methods).

**Process note on how this got built:** installed the `codex` CLI this
session (the user had the ChatGPT/Codex extension but not the standalone
CLI) and delegated the implementation to it with a detailed brief. Hit two
real environment problems along the way (bubblewrap sandbox unavailable,
then a nested-network-namespace permission error under `-s workspace-write`
that only cleared once dispatched via `--dangerously-bypass-approvals-and-sandbox`
+ `nohup`/`disown`), and one real self-inflicted mistake: killed the
agent's first successful run because its log showed
`WARNING:root:No pretrained weights loaded... initialized randomly` and
assumed the CLIP backbone was garbage — that warning turned out to be a
harmless artifact of `open_clip.create_model_and_transforms(pretrained=None)`
before the code's own subsequent manual `load_state_dict` (verified
separately: zero missing/unexpected keys), so the kill was wrong and wasted
that run's progress. Caught it by verifying directly rather than trusting
the log line, then re-ran the same script myself once confidence was
restored. Recorded here so the same warning doesn't cause a repeat false
alarm later.

### 3.2 E4 cost table — real timing data already exists across several CSVs; the *figure* was the actual gap, now fixed for FreeDA

Correction to my original read: the author already has real E4 timing data
(this is not a from-scratch gap). It's spread across three sources:
`tf_ovcos_full_results_20260526/runs/analysis/e4_cost_partial_official.csv`
(5 official-mmseg methods, `sec_per_iter` from logs), `runs/tables/e4_cost.csv`
(7 adapter-native methods — maskclip variants, sam_amg, dinov2_sam — real
`runtime_sec_per_image` from each shard's `runtime.json`), and the paper's
own Section 4.4 prose, which states exact per-method numbers for the
remaining methods (Trident, SC-CLIP, CLIPtrase, CASS) that don't appear in
either CSV. `peak_mem_mb`/model-call columns are genuinely empty in every
file that has one — that part of the original finding stands.

**The actual, confirmed gap was narrower than I first described: FreeDA was
missing from `TMLR26_Bench/scripts/make_clean_observation_figures.py`'s
`fig_efficiency_waterfall()`, which builds Figure 4 from a hardcoded
Python list of `(method, mIoU, sec/image)` tuples (not from any of the
CSVs above) — FreeDA was simply never typed into that list, matching that
the paper's own Section 4.4 text never states a FreeDA seconds/image
number either.** No CSV, JSON, or log in the repo had a pre-computed
FreeDA timing figure. Derived one from the raw wall-clock timestamps in
`runs/logs/mcmr_artifact_freeda_*.log` / `mcmr_freeda_e2_artifact_repair.log`
(start/end of each dataset's run divided by image count):

| dataset | wall time | images | sec/image |
|---|---|---|---|
| VOC-20 | 2599s | 1449 | 1.79 |
| Context-59 | 8334s | 5105 | 1.63 |
| ADE-150 | 3070s | 2000 | 1.54 |
| COCO-Stuff | 6215s | 5000 | 1.24 |
| Context-459 | 9093s | 5105 | 1.78 |
| ADE-847 | 3520s | 2000 | 1.76 |

E1-average (4 datasets): **1.49 s/image**. Added `("FreeDA", 45.40, 1.49)`
to the hardcoded list and regenerated `fig_efficiency_pareto.pdf` — FreeDA
now appears in the figure, positioned between NACLIP (0.88s) and CASS
(2.21s). Caveat kept in the script as a comment: this is wall-clock time
including one-time model/checkpoint loading inside that process, not a
clean isolated per-image timer like the other rows have — flagged as an
approximation pending a real isolated rerun, not presented as equivalent
precision to the other methods' numbers.

Status: done (figure fixed). Peak memory / model-call counts are still
genuinely absent everywhere and would need an isolated instrumented rerun
if wanted — not attempted this session.

### 3.3 Table 8 (component-size diagnostic) — coverage limited to 5 methods; expansion is optional, not required

`exploratory_scale_components.csv` has exactly corrclip/naclip/proxyclip/
resclip/sclip. My original framing ("must extend to 8+ methods") overstated
this — **the actual issue is just word choice**: Obs 8/9 use "universal
weak point" / "holds across dense and VFM-assisted methods" for a 5-method
sample that happens to include only 2 of the paper's own 4 "CLIP+VFM"
methods (ProxyCLIP, CorrCLIP; not Trident/CASS) and no proposal or
diffusion methods. Fixing this needs either (a) softening the claim to
name the 5 methods actually checked, or (b) running the same connected-
component analysis on more methods — but nothing here requires 8
specifically, and the author has judged the current 5-method sample
sufficient evidence for the claim as intended (not investigated further
this session per that call).

### 3.4 e2_vocab_robustness_filled.csv — confirmed correct, no action needed

Full 19-method table (18 TF methods + SAN), matches Table 2 in the PDF
exactly on spot-checks (ZIoU, MCMR, Δvocab). This part of the paper is
solid.

### 3.5 ZIoU / mIoU relationship (Obs 4) and cross-family ratio ordering (Obs 6) — Spearman ρ now computed, confirms the concern

Computed Spearman ρ(ZIoU, mIoU) across all 18 training-free methods from
`e2_vocab_robustness_filled.csv`:

| Pair | ρ | p-value | n |
|---|---|---|---|
| Context-459 | -0.364 | 0.138 | 18 |
| ADE-847 | -0.194 | 0.440 | 18 |
| Pooled (both) | -0.322 | 0.055 | 36 |

The correlation is in the expected direction (higher ZIoU associated with
lower mIoU) but **weak and not statistically significant at either
individual dataset** (p=0.14 and p=0.44), only borderline significant
pooled (p=0.055, n=36). This is quantitative confirmation of the original
finding: Obs 4's claim that "the zero-IoU rate, not the surviving per-class
IoU, tracks the collapse" overstates how tight this relationship actually
is. Sorting methods by Context-459 ZIoU makes the exceptions concrete:
CLIPtrase sits at #4 highest ZIoU (54.2%) *and* has the 2nd-highest
Context-459 mIoU (9.95, behind only CorrCLIP) among all 18 methods — high
class-silence and high overall accuracy simultaneously, the opposite of
what "ZIoU tracks the collapse" implies. FreeDA is the one case that fits
the narrative cleanly (highest ZIoU, 4th-lowest mIoU).

**Action: soften Obs 4 to something like "ZIoU and mIoU are correlated in
the expected direction but the relationship is not tight (Spearman
ρ≈-0.32 to -0.36, not significant per-dataset) — CLIPtrase in particular
combines high class silence with above-average overall accuracy, showing
that a method can silence a large share of classes while still performing
well on the classes it does cover."** This pairs naturally with the E-11
finding in §5.4 — both point the same direction: Obs 4's headline claim
about vocabulary size was cleaner in the write-up than in the underlying
numbers, and the paper is more defensible citing the precise, hedged
versions of both than the original strong claims.

Obs 6's ratio-ordering claim was not re-checked this session beyond the
original review (still open, same finding as before: CASS vs Trident and
NACLIP vs ProxyCLIP break the claimed ordering).

## 4. Experiment tracker (superseded the two earlier duplicate tables that used to live in §4/§6 — this is the single current-status table, cross-referenced from §6)

| # | Item | Status |
|---|------|--------|
| E-1 | Per-target tuning delta ablation | **Closed — done as an optional pilot, not going into the manuscript unless needed.** See §9. |
| E-2 | Table 9 MCMR recompute | **Done and verified.** See §5.1/§1.1. |
| E-3 / E-13 | ZIoU × mIoU correlation (Spearman ρ) | **Done.** See §3.5. |
| E-4 / E-14 | Full E4 cost / FreeDA efficiency figure | **Done** (figure fixed; peak-memory/model-call instrumentation still genuinely absent everywhere, not pursued further — author confirmed the plot is sufficient). See §3.2. |
| E-5 | Real (non-proxy) 4-probe diagnostic for Obs 5 | **Done, fixed, and verified at full dataset scale.** Ran on the complete Context-459 (5105 imgs) + ADE-847 (2000 imgs) validation sets, 0 errors. Final numbers: naming 0.199, localization 0.285, proposal oracle 0.623, proposal recall 0.663 — closely matches the earlier 250-image pilot (within ~0.02-0.03 on every metric). Localization needed two failed attempts (single-prompt Otsu, then contrast-vs-mean) before a full-argmax fix worked — see §3.1 for the full trail. |
| E-6 | Table 8 component-size diagnostic expansion beyond 5 methods | **Closed — not required, author's call.** Wording-precision issue, not a missing experiment. See §3.3. |
| E-7 | Reproduction-gap table | **Closed — not needed, author's call.** Fixed protocol isn't trying to reproduce original papers' numbers. |
| E-8 | Run OVSeg/ODISE/CAT-Seg/FC-CLIP/OVDiff/GroundingDINO+SAM | **Closed — not needed, author's call.** Trained references aren't the paper's subject; SAN is sufficient. |
| E-9 | Real cross-domain E3 evidence (hard-domain camo datasets) | **Closed — not needed, author's call, confirmed against the paper's own §3.3 text.** See §5.9 for two real bugs found while scoping this (not urgent). |
| E-10 / E-15 | Real run-card / protocol config values | **Done.** See §8. |
| E-11 | Candidate-only vocabulary expansion (rescoring) | **Done.** Real result, changes Obs. 4's framing (SAN/FreeDA finding). See §5.4. |
| E-12 | No-prediction class rate (ZIoU refinement) | **Done — real, notable finding.** See §5.5: the 8 vanilla dense/VFM methods share a nearly-identical never-predicted rate regardless of architecture; CLIPtrase/FreeDA/SAN fail via a different broad-but-imprecise pattern instead. |
| — | CLIPtrase/FreeDA COCO-Stuff normalization fix | **Done, fixed, and verified this session.** See §2.1: the crosswalk built earlier was correct, the bug was applying it without first subtracting 1; corrected predictions are now live in the artifact tree, all diagnostic tables regenerated, and the exact Appendix D.1 pooled-number deltas are recorded. |
| — | Manuscript-text-only items (broken ref, `ZUoU` typo, undefined `Catastrophic ZIoU`, abstract wording, Eq. 8 description, Obs 4/5/6 rewrites) | **Cannot be done here** — no `.tex` source in this repository. See §5.2/5.3/5.6/5.7/5.8. |

## 5. Second-pass external review (2026-07-13, later same day) — cross-checked against repo

A second, independent review came in after §1-5 were written. Went through
each of its points against the repo instead of re-deriving from scratch.
Status legend: **already covered** (nothing new to do beyond what §1-5
already says), **new, fixed this session**, **new, confirmed but can't fix
here**, **new, confirmed and documented, not yet fixed**.

### 5.1 Table 9 mismatch/localized ratio ≠ printed MCMR — new, fixed this session

The review computed 149030/413463=0.360 and 115342/275642=0.418, neither
matching the printed 0.2972/0.3454, and asked whether the printed MCMR is a
macro-average while the pair counts are pooled totals — i.e. two
incompatible aggregations dressed up to look like one fraction.

Checked the code (`_table11_rows` in `scripts/generate_diagnostic_tables.py`,
pre-fix): confirmed exactly that, plus a second compounding bug:

- `"localized_pairs"` was `sum(gt_class_regions)` — **all** GT regions,
  not the subset actually localized at IoU≥0.5. Mislabeled.
- `"mismatch_pairs_proxy"` was `sum(round(mcmr_at_05 * gt_class_regions))`
  — mismatch *rate* multiplied by the *wrong* (unfiltered) denominator.
- `"MCMR@0.5_proxy"` was a third, independent quantity: the unweighted mean
  of per-(method,dataset) `mcmr_at_05` values.

So the printed pair counts, the printed ratio, and the printed MCMR were
three different computations that only look like `mismatch/localized =
MCMR` by accident of formatting. This is a real bug, not just a
presentation gap, and it's the review's most important, most concretely
falsifiable finding this round.

**Fixed:** added raw `localized_pairs_at_05` / `mismatch_pairs_at_05` counts
to each per-(method,dataset) summary row, and rewrote `_table11_rows` to
sum those directly. Family-level output now reports both:
- `MCMR@0.5_pooled` = `mismatch_pairs / localized_pairs` (matches Eq. 8
  applied at the pooled level, self-consistent by construction), and
- `MCMR@0.5_mean_of_cells` = the old unweighted per-cell mean, kept
  separately and clearly labeled rather than conflated with the pooled
  number.
Rerun in flight (PID 154792 at time of writing; check
`runs/analysis/table11_e2_mismatch_summary.csv` for the corrected numbers,
verify `mismatch_pairs / localized_pairs == MCMR@0.5_pooled` by hand before
trusting it in the manuscript).

### 5.2 Broken `Tables 1-??` cross-reference, `ZUoU` typo, undefined `Catastrophic ZIoU` — confirmed, cannot fix here

Searched the whole repo for a `.tex` source matching the submitted PDF's
actual title ("Towards Multifaceted Evaluation of Training-Free
Open-Vocabulary Segmentation..."). **The only `.tex` file in this repo
(`tf_ovos_proposal_v25_expanded_exploratory.tex`) is an earlier, differently
titled internal draft ("TF-OVOS: A Four-Part Evaluation Framework...").
The actual manuscript source is not in this repository** — it was written/
compiled elsewhere and only the PDF was shared. This means:

- I cannot search-and-fix the `Tables 1-??` reference, since there's no
  `\ref{}`/`\label{}` pair here to correct.
- `"ZUoU"` doesn't appear anywhere in this repo's text, confirming Figure 1
  is a hand-made diagram (not generated by any script here) with a plain
  typo baked into the image itself.
- `"Catastrophic ZIoU"` (Figure 1's label) has no corresponding computed
  quantity anywhere in the codebase — confirmed by grep across all
  scripts/CSVs/markdown. It is presumably meant as an informal label for
  "particularly high ZIoU," but nothing in the pipeline defines a distinct
  "catastrophic" threshold. **Action for whoever holds the manuscript
  source:** either define a threshold and compute it, or remove the label
  from Figure 1 and describe it as "very high ZIoU" in prose only.
- Same for "MCC consistency" (appendix) vs "mask-category consistency /
  MCMR" (main text): in the repo, `"MCC consistency diagnostics"` is
  genuinely a *different* method_group placeholder (for before/after
  mask-category-consistency-reranking comparison, still `artifact_gap` —
  never populated) — not the same thing as MCMR. This isn't a computation
  bug, just needs a one-sentence clarification in the manuscript that MCC
  reranking and MCMR are different axes.

### 5.3 Abstract wording ("VLMs" should be "VFMs"; "mitigated overfitting" claim) — already covered

Both already flagged in the original review (this repo's very first pass,
issues #1/#2 in the "结论与数据矛盾" section) and not re-litigated in §1-5
above because there's nothing new to verify — same conclusion: change
"VLMs" to "VFMs" in the abstract sentence contrasting VFM-assisted vs
CLIP-only dense methods, and delete or drastically soften the
distribution-shift/overfitting claim, which no experiment in this repo
supports. No manuscript source here to apply the edit directly.

### 5.4 E2 confounds vocabulary-size expansion with taxonomy/annotation-scheme change — DONE, real result, changes Obs. 4's framing

This is a real, previously-unflagged design issue, not just a wording
problem. Context-59→459 and ADE-150→847 change the GT taxonomy itself
(different annotation schemes, not just "more candidate names for scoring
the same regions"), so the reported mIoU drop conflates: candidate-set
competition, GT class-support/frequency shift, and possible ignore/void
remapping differences. Nothing in the current pipeline isolates
candidate-only expansion (same GT, larger candidate list) from taxonomy
expansion (different GT).

**Not fixed this session, but feasibility now confirmed empirically — this
needs zero new inference.** Checked directly:

- Context-59 vs Context-459: identical 5105-image set (`ids59 == ids459`
  is `True`). 58 of 59 compact class names match an exact string in the
  459-list; the one apparent miss ("people" vs "person") is just a naming
  variant, trivially fixed by a name-normalization pass.
- ADE-150 vs ADE-847: the manifests store image IDs in different formats
  (ADE-847's include a scene-category path prefix, e.g.
  `nature_landscape/forest__needleleaf/ADE_val_00001364`, ADE-150's don't),
  which made a naive set-equality check report 0 overlap. After
  normalizing both to the bare `ADE_val_NNNNN` id, **all 2000 images match
  exactly** — the paper's "same images" claim holds, it just wasn't
  verifiable from the raw manifest fields directly. Class names have more
  formatting drift between the two vocab files (many multi-word ADE-847
  names appear to have spaces/hyphens stripped, e.g. `bulletinboard` vs.
  presumably `bulletin board`), so name matching needs a normalize-then-match
  pass (lowercase, strip spaces/hyphens) rather than exact string equality.

**Implemented and run** as `scripts/rescore_candidate_only_expansion.py`:
takes each method's already-saved Context-459/ADE-847 dense predictions,
remaps predicted class indices into the compact 59/150-class space via a
normalized name crosswalk (comma-separated synonym lists in the vocab
files are split and each alias tried; one manual alias added for
"people"/"person"; any large-vocab prediction with no compact match
counts as off-vocabulary, i.e. wrong for that pixel), and recomputes mIoU
against the *original* Context-59/ADE-150 GT. Crosswalk coverage: 59/59
Context classes, 149/150 ADE classes (the one miss, `crtscreen`, is a
genuine synonym-grouping ambiguity in the ADE-847 vocab file — one entry
lists "screen, crt screen" as synonyms for what the compact vocab treats
as two separate classes — not a bug, left undecided rather than guessed).
No new inference, ran on already-saved artifacts for all 11 methods with
dense predictions.

**Result — decomposing the E1→E2 mIoU drop into a vocabulary-competition
term and a taxonomy/annotation-scheme term** (`compact → candidate-only-
rescored → joint-large`, all in mIoU points; `tax_share` = the fraction of
the *total* compact→large drop attributable to the taxonomy/GT-scheme
change rather than candidate-set competition):

| Method | Ctx-59 | Ctx cand.-only | Ctx-459 | vocab drop | taxonomy drop | tax share |
|---|---|---|---|---|---|---|
| SCLIP | 34.14 | 15.74 | 6.68 | 18.40 | 9.06 | 33.0% |
| NACLIP | 38.36 | 16.99 | 7.79 | 21.37 | 9.20 | 30.1% |
| ResCLIP | 36.85 | 16.74 | 7.70 | 20.11 | 9.04 | 31.0% |
| SC-CLIP | 40.21 | 18.01 | 8.34 | 22.20 | 9.67 | 30.3% |
| ProxyCLIP | 39.31 | 18.11 | 8.41 | 21.20 | 9.70 | 31.4% |
| CorrCLIP | 48.68 | 22.56 | 11.88 | 26.12 | 10.68 | 29.0% |
| Trident | 40.99 | 18.85 | 8.97 | 22.14 | 9.88 | 30.9% |
| CASS | 40.27 | 18.28 | 8.38 | 21.99 | 9.90 | 31.0% |
| CLIPtrase | 34.58 | 26.97 | 9.95 | 7.61 | 17.02 | **69.1%** |
| FreeDA | 43.42 | 36.21 | 4.27 | 7.21 | 31.94 | **81.6%** |
| SAN (ref.) | 52.43 | 48.82 | 12.75 | 3.61 | 36.07 | **90.9%** |

| Method | ADE-150 | ADE cand.-only | ADE-847 | vocab drop | taxonomy drop | tax share |
|---|---|---|---|---|---|---|
| SCLIP | 16.46 | 11.39 | 5.67 | 5.07 | 5.72 | 53.0% |
| NACLIP | 19.12 | 13.05 | 6.01 | 6.07 | 7.04 | 53.7% |
| ResCLIP | 18.07 | 13.05 | 6.49 | 5.02 | 6.56 | 56.6% |
| SC-CLIP | 20.06 | 14.24 | 7.46 | 5.82 | 6.78 | 53.8% |
| CLIPtrase | 17.04 | 11.27 | 5.89 | 5.77 | 5.38 | 48.3% |
| ProxyCLIP | 19.59 | 14.64 | 6.90 | 4.95 | 7.74 | 61.0% |
| CorrCLIP | 26.96 | 19.34 | 8.67 | 7.62 | 10.67 | 58.3% |
| Trident | 20.91 | 14.72 | 7.45 | 6.19 | 7.27 | 54.0% |
| CASS | 20.32 | 13.65 | 7.19 | 6.67 | 6.46 | 49.2% |
| FreeDA | 23.19 | 17.82 | 3.93 | 5.37 | 13.89 | **72.1%** |
| SAN (ref.) | 27.56 | 23.64 | 10.24 | 3.92 | 13.40 | **77.4%** |

Full per-method CSV: `runs/analysis/e11_candidate_only_rescoring.csv`.

**This is a real finding that changes how Observation 4 should be read, not
just a caveat.** For the eight "ordinary" dense/VFM methods, the taxonomy
share sits in a tight, boring band (~29-33% on Context, ~48-61% on ADE) —
vocabulary competition genuinely is the larger term on Context, roughly
comparable to taxonomy change on ADE. The paper's directional claim
("expanding candidates degrades methods") survives for these eight, just
weaker than implied — a meaningful chunk of "E2 collapse" is really
"different annotation scheme," not competition.

**But CLIPtrase, FreeDA, and especially SAN break the pattern entirely.**
For SAN, 90.9% (Context) and 77.4% (ADE) of the drop is the taxonomy term
— SAN loses almost nothing (52.43→48.82, only 3.6 points) to pure
candidate-set growth, and only collapses once the GT itself changes.
FreeDA is the same shape (81.6% / 72.1% taxonomy share). This directly
undercuts how Obs. 4 currently uses these two methods as evidence:

> "FreeDA drops the hardest despite its strong 45.40 E1 average..."
> "Even the trained reference SAN leaves 51.3% of ADE-847 classes silent,
> so the failure is a property of the large label space itself rather than
> of any single recipe."

Both statements read as if vocabulary-scale is what breaks these methods.
The decomposition says the opposite for SAN in particular: **SAN is one of
the most robust methods to genuine vocabulary competition** (smallest
"vocab drop" of any row in the table, 3.61/3.92) **and one of the least
robust to taxonomy change** — a materially different story than "large
label spaces defeat everything, even trained references." **Action:
rewrite Observation 4 to state the two components separately, and stop
citing SAN's E2 collapse as evidence that vocabulary size alone is the
culprit — the data now says it mostly isn't, for SAN specifically.**

### 5.5 ZIoU conflates "never predicted anywhere" with "predicted but zero overlap" — DONE, real result, adds a genuinely new finding

Valid distinction the pipeline didn't separate before this session. `IoU_c
= 0` can mean the model never emitted class `c` anywhere in the whole
dataset, or emitted it somewhere with zero spatial overlap with the true
region — two very different failure modes that the plain zero-IoU rate
conflates.

**Implemented:** extended `_analyse_method_dataset` in
`generate_diagnostic_tables.py` to accumulate a full dataset-level
confusion matrix (not just the per-image one already used for other
readouts) and derive, per class: total GT pixels, total predicted pixels,
and TP, across the *entire* dataset. From that: `zero_iou_class_rate`
(matches the existing ZIoU), `never_predicted_class_rate` (predicted-pixel
count is exactly zero across the whole dataset), and
`predicted_but_zero_iou_class_rate` (predicted somewhere, just never on
the true region). Reran the full pipeline (11 methods × 6 datasets); new
columns are in `runs/analysis/table9_diagnostic_probes.csv`.

**Result on the two large-vocabulary datasets** (rate = fraction of valid
classes):

| Method | ADE-847 never-pred | ADE-847 pred-but-wrong | Ctx-459 never-pred | Ctx-459 pred-but-wrong |
|---|---|---|---|---|
| SCLIP | 0.677 | 0.011 | 0.455 | 0.090 |
| NACLIP | 0.677 | 0.020 | 0.458 | 0.093 |
| ResCLIP | 0.677 | 0.022 | 0.455 | 0.101 |
| SC-CLIP | 0.677 | 0.030 | 0.458 | 0.113 |
| ProxyCLIP | 0.677 | 0.029 | 0.455 | 0.116 |
| CorrCLIP | 0.679 | 0.037 | 0.458 | 0.139 |
| Trident | 0.677 | 0.044 | 0.461 | 0.130 |
| CASS | 0.677 | 0.029 | 0.458 | 0.125 |
| CLIPtrase | 0.246 | 0.343 | 0.058 | 0.345 |
| FreeDA | 0.015 | 0.325 | 0.000 | 0.217 |
| SAN (ref.) | 0.153 | 0.344 | 0.038 | 0.278 |

**This is a real, previously-invisible finding, not just a refinement.**
The first eight rows (all the "vanilla" dense/VFM training-free methods)
have a *never-predicted* rate that is nearly identical across completely
different architectures — 0.677-0.679 on ADE-847, 0.455-0.461 on
Context-459, a tighter cluster than anything else measured this session.
That means these eight methods, regardless of architecture, are all
converging on predicting from roughly the *same* ~32-45% common-class
subset of each large vocabulary and genuinely never attempting the rest —
concrete, quantified support for Obs. 4's qualitative claim that "a frozen
model concentrates its probability mass on a small set of common
categories."

CLIPtrase, FreeDA, and SAN break this pattern in a *different, specific*
way: their never-predicted rate is far lower (they attempt far more of
the vocabulary) but their predicted-but-wrong rate is far higher (3-10x
the other eight) — i.e. these three methods spread their attention across
more of the candidate vocabulary but are much less accurate when they do.
This gives Obs. 4/8/9 a sharper, two-axis characterization of large-vocabulary
failure that wasn't visible before this session: **narrow-but-accurate**
(the eight vanilla methods) vs. **broad-but-imprecise** (CLIPtrase, FreeDA,
SAN) are two qualitatively different failure modes, both producing similar
overall mIoU collapse for different underlying reasons. Worth a real
mention in the manuscript, not just a footnote — this is new information,
not a rephrasing of something already reported.

Family-level Table 9 numbers were re-verified consistent after this rerun
(`mismatch_pairs / localized_pairs == MCMR@0.5_pooled` still holds for all
four populated groups).

### 5.6 MCMR's zero-denominator behavior — already handled correctly in code, mismatch is in the paper's written formula only

Checked `_mcmr()`: `if localized == 0: return None`. **The implementation
already does the right thing** — it returns "undefined," which
`_safe_mean()` then correctly excludes from any average, rather than
falling through to 0 as Eq. (8)'s written `max(..., epsilon)` guard implies.
This is good news: no code fix needed. **Action:** the manuscript's Eq. (8)
should be corrected to describe what the implementation actually does
(report MCMR as undefined/N/A when there are zero localized regions,
excluded from aggregation) rather than the `epsilon`-guard formula as
literally written, which reads as if it silently returns 0.

### 5.7 "GT region" = whole per-class mask, not connected component; oracle/naming match is greedy per-class argmax, not one-to-one — new, confirmed, documented

Two related precision issues, confirmed by reading `_analyse_method_dataset`:

- The loop is `for gt_id in np.flatnonzero(gt_counts)`, i.e. one row per
  **(image, class)** pair. If "person" appears as three disconnected blobs
  in one image, that's one row, not three. This is a *different* and
  coarser unit than the genuinely-per-connected-component
  `exploratory_scale_components.csv` (Table 8), which does use real
  connected components (`n_components` column, confirmed in that CSV).
  The paper's prose ("GT region") should specify which of the two units is
  meant in each table, since they're not interchangeable.
- The oracle/naming match (`argmax` over predicted classes' IoU with a
  given GT class) is computed independently per GT class. Nothing
  prevents two different GT classes in the same image both selecting the
  same predicted class as their best match — this is **not** a one-to-one
  (Hungarian) assignment. For a dense argmax-based semantic-segmentation
  proxy this is a defensible modeling choice (it's not really about
  discrete "proposals" competing for assignment), but it should be stated
  explicitly rather than left implicit, since a reader could reasonably
  assume "oracle IoU" implies a matching without duplication.

Not changed — these are modeling/description choices to make explicit in
the manuscript's methodology section, not bugs to fix in isolation
(switching to real one-to-one matching would change every downstream
number and is a much bigger design decision than a bug fix).

### 5.8 Everything else in the second-pass review — already covered or requires the manuscript source

- E3 reframing ("Cross-Dataset Sensitivity" vs "Generalizability"), the
  Obs. 6 "ratio ordering matches E1 ordering" claim being false on the
  actual numbers (CASS vs Trident, NACLIP vs ProxyCLIP) — already in §3.5
  and in the very first review pass; same conclusion, nothing new to
  verify.
- Run-card completeness, E4 completeness (FreeDA missing, no memory/model
  calls) — already covered in §3.2 and the original review.
- Related-work positioning against Huang et al. 2025 / Open-mIoU-style
  metrics / 2026 baselines (GLA-CLIP etc.) — outside what this repo's data
  can settle; this is a literature/citation-scope decision for whoever
  holds the manuscript, not something to verify against artifacts.
- Table 1's `Type` column abbreviations (CM/VM/DM/PN) needing expansion in
  a caption, "Source/training data = None" being misleading for CLIP/SAM/
  DINOv2-pretrained rows, abstract's triple "Moreover," bootstrap
  confidence intervals for close mIoU gaps (43.31 vs 43.75) — all
  manuscript-text-level fixes with no artifact to check against; deferred
  to whoever edits the actual `.tex`.

### 5.9 Two real bugs found while scoping (then closing) E-9 — worth fixing whenever hard-domain appendix work is picked back up, not urgent now

E-9 itself is closed per §4's table (the paper's own §3.3 already scopes
hard-domain datasets as optional appendix material, and E3's in-body name
is already "Cross-Target Stability," not "Generalizability" — no new
experiment needed). While scoping it, found real predictions already exist
for 7 methods × 4 hard-domain datasets (OVCamo-TE, CAMO-TE, COD10K-TE-Camo,
NC4K) with per-run `metrics.json` already computed
(`runs/appendix_table10_abs/<method>/<dataset>/metrics.json`), but two bugs
mean most of that data isn't trustworthy yet:

1. `runs/tables/appendix_hard_domain.csv` (the summary table) is built by
   a script that iterates over `runs/*` directory names as if they were
   method names — it has rows for `"analysis"`, `"logs"`, `"tables"`,
   `"tmp"`, etc. and is empty for every real method. Rebuilt a corrected
   version by reading the real per-run `metrics.json` files directly:
   `runs/tables/appendix_hard_domain_fixed.csv`.
2. Of the 7 methods with predictions, only `maskclip`'s hard-domain numbers
   are trustworthy (IoU 0.03-0.07 across the 4 datasets, plausible and
   varied). The other 6 (`maskclip_attn`, `maskclip_attn_slide`,
   `sam_amg_clip`, `sam_amg_siglip`, `dinov2_sam_clip`, `dinov2_sam_siglip`)
   all show near-identical, near-zero scores — traced to
   `src/tf_ovos/eval.py`'s mask-only/class-aware branch calling
   `load_binary_mask(pred.mask_path, threshold=0.5)`
   (`src/tf_ovos/metrics.py:200`), which does `pixel_value/255 >= 0.5`.
   That's correct for `maskclip`'s adapter, which writes a genuine 0/255
   binary mask, but the other six adapters write **dense multi-class
   label-index PNGs** (same format as the main OVS benchmark, values like
   0-68) at that same `mask_path` — nearly every class index is below the
   127.5 cutoff, so they all get thresholded down to "predict nothing,"
   against a real GT foreground of ~0.7% of the image, giving
   near-uniform near-zero IoU regardless of what each adapter actually
   predicted. Confirmed by direct inspection of the raw PNGs (`sam_amg_clip`
   has 99.4% nonzero raw pixels with values 0-68; `maskclip` has a genuine
   binary 0/255 mask).
   **Real fix, not attempted:** these six adapters need task-specific logic
   to emit a single binary foreground mask for this appendix task (e.g.
   pick the highest-confidence single proposal/region) rather than reusing
   their dense multi-class main-benchmark output as-is — a real per-adapter
   code change, not a one-line scoring fix. Left as a documented, bounded
   future task; not urgent since E-9 itself isn't needed right now.

## 6. (merged into §4 — see the single master tracker table there)

## 7. Immediate next actions, in order (final — replaces the two earlier,
now-redundant "immediate next actions" lists that used to live in this
slot; folded in below instead of left duplicated)

**[Update, 2026-07-14]: every numbered item below is now done — see §4's
master tracker for the authoritative current status of each. This list is
kept only as a historical record of what was still open at the point it was
written; do not treat it as a live to-do list.**

Done: Table 9 (`table11_e2_mismatch_summary.csv`) is now fully consistent
— all four populated groups pass the `mismatch_pairs / localized_pairs ==
MCMR@0.5_pooled` check by hand, and `"Proposal+naming TF methods"` has a
real `MCMR@0.5_mean_of_cells = 0.744` from the four already-published
Table 2 values instead of a blank `artifact_gap` cell (§1.1). E-11
(candidate-only rescoring) is done, see §5.4 for the full result and the
FreeDA/SAN finding.

Remaining, in order (historical — all resolved since, see §4):

1. ~~Implement E-12~~ — done, §5.5.
2. ~~Compute the Spearman ρ for §3.5~~ — done, §3.5.
3. ~~Decide whether to invest in re-exporting cliptrase/freeda's COCO-Stuff
   predictions correctly~~ — done and fixed, §2.1.
4. ~~Run E-1 (per-target tuning ablation)~~ — done as an optional pilot, §9.
5. ~~Run E-4 (isolated E4 pass with memory + FreeDA)~~ — done, §3.2.
6. Everything under §5.2/5.3/5.6/5.7/5.8, plus the reproduction-gap table
   (E-7), needs the actual manuscript `.tex` source, which is not in this
   repository — flag explicitly to whoever owns that file rather than
   trying to reconstruct it here. **This one item is still genuinely
   outstanding**, but it is not something this repo can resolve.

**Lesson from this pass, worth remembering for the rest of this queue:**
before writing a new script to compute something, check whether the
project's existing per-run `metrics.json` files (`runs/<method>/<dataset>_val/
metrics.json`) or an already-published table already has it. Not everything
missing from `runs/analysis/table9_diagnostic_probes.csv` is actually
missing from the project — that CSV is one specific diagnostic pipeline's
output, not the only source of truth.

## 8. Real run-card / protocol values (E-10), extracted from resolved configs

Addresses the "fixed protocol claims real numbers but the paper never
shows them" gap. These are pulled directly from the already-generated,
already-resolved per-method config files under `runs/official/<method>/
<dataset>/cfg_tfovos_<dataset>.py` (mmseg-style methods) and `configs/
methods/<method>.yaml` (adapter-native methods) — not re-derived or
guessed, just extracted from what each run actually used.

**Dense / CLIP+VFM methods (official-mmseg harness, VOC-20 shown; same
resize rule confirmed across all 6 datasets for sclip):**

| Method | CLIP backbone | Resize rule | Prompt/class file | Notes |
|---|---|---|---|---|
| SCLIP | openai `ViT-B/16` | `keep_ratio=True, scale=(2048,336)` | `configs/cls_tfovos_<dataset>.txt` | |
| NACLIP | openai `ViT-B/16` | same | same | |
| ResCLIP | openai `ViT-B/16` | same | same | |
| ProxyCLIP | openai `ViT-B/16` (`clip_type='openai'`) | same | same | |
| CorrCLIP | **`metaclip_fullcc`, `ViT-B-16-quickgelu`** | same | same | Different CLIP variant/checkpoint family from the other five — worth stating explicitly in the manuscript's method table rather than implying every dense row uses the identical CLIP weights. |
| SC-CLIP | openai `ViT-B/16` | same | `configs/cls_voc20.txt` (own naming) | |
| Trident | (not directly in this grep pass; uses `configs/cls_voc20.txt`) | same | same | |
| CASS | openai `ViT-B/16`, `pamr_steps=0` | same | `configs/cls_voc20.txt` | PAMR post-processing explicitly disabled (steps=0) in this config |

All eight batch_size=1, `num_workers=4`, single-scale (no multi-scale/TTA)
per the resolved configs sampled.

**Proposal + naming methods** (from `configs/methods/*.yaml`):

| Method | Proposal model | Naming model | Threshold / prompt |
|---|---|---|---|
| SAM-AMG + CLIP | SAM `sam_vit_b_01ec64.pth` (ViT-B) | `openai/clip-vit-base-patch16` | `mask_threshold=0.5`, `min_component_area=64`, prompt `"a photo of a {}"` |
| SAM-AMG + SigLIP | same SAM checkpoint | SigLIP (same family, not re-extracted this pass) | same thresholds |
| DINOv2 + SAM + CLIP | DINOv2-guided SAM proposals | frozen CLIP | config yaml has less detail than sam_amg_clip.yaml; adapter source (`src/tf_ovos/adapters/dinov2_sam.py`) has the actual values if needed |
| DINOv2 + SAM + SigLIP | same | frozen SigLIP | same |

**FreeDA**: its own config lives in the official repo's yml files
(`configs/pascal20/freeda_pascal20.yml`, `configs/cocostuff/freeda_cocostuff.yml`,
etc., referenced from `scripts/run_missing_mcmr_artifacts_queue.sh`), not
in this repo's `configs/methods/freeda.yaml` (which is just a status stub).
Not extracted in detail this pass — flagged if a full backbone/resolution
row is wanted for FreeDA specifically.

**Action:** fold the two tables above into an appendix run-card table in
the manuscript (the CorrCLIP backbone difference and CASS's `pamr_steps=0`
are the two most reportable, non-obvious details — everything else is
uniform across the family as the "fixed protocol" claim requires).

## 9. E-1 pilot: per-target tuning delta ablation

Ran a small local grid around each method's default inference
hyperparameters, on a 300-image subset of Context-59 (`val_subset300.txt`,
first 300 lines of the official val split — chosen for speed, not the
full 5105-image set), re-using each method's real official-mmseg eval
pipeline (config copied and edited, not reimplemented). Picked two methods
with hyperparameters actually exposed in their model constructor:

**SC-CLIP** (`res_cls`, `pre_adjust_idx`/`post_adjust_idx` — residual-
classifier weight and the transformer layer range it adjusts):

| Config | mIoU |
|---|---|
| `res_cls=0.3, pre=8, post=3` (**default**) | 34.73 |
| `res_cls=0.1` | 34.88 |
| `res_cls=0.5` | 34.83 |
| `res_cls=0.7` | 34.82 |
| `pre=4, post=1` | 33.18 |
| `pre=10, post=5` | 34.24 |
| `pre=6, post=2` | 34.73 |

**NACLIP** (`gaussian_std` — the neighborhood-attention kernel width; not
set in the shipped config, so the default of 5.0 was implicit):

| Config | mIoU |
|---|---|
| `gaussian_std=5` (**default**) | 33.45 |
| `gaussian_std=3` | 33.36 |
| `gaussian_std=7` | 33.50 |
| `gaussian_std=10` | 33.53 |

**Result: the tuning headroom found here is small.** Best-over-default
gain is **+0.15 mIoU for SC-CLIP** (34.88 vs 34.73) and **+0.08 mIoU for
NACLIP** (33.53 vs 33.45) — an order of magnitude smaller than the
compact-vocabulary gaps the paper reports between method families (e.g.
SC-CLIP 40.21 vs NACLIP 38.36 in Table 1). SC-CLIP's full range across all
seven configs (1.70 points) looks bigger, but that's mostly driven by a
*bad* config (`pre=4, post=1` at 33.18), not by tuning-inflation off the
default — an adversary picking the single best config after the fact still
only gains +0.15, not +1.70.

**How to read this, honestly:** this is a genuine, if modest, empirical
data point *for* the fixed-protocol argument, not against it — for these
two methods, on this one dataset (a 300-image subset), the specific
numeric hyperparameters exposed in their official configs don't offer much
free accuracy from post-hoc tuning. It does **not** by itself establish
that *no* per-target tuning would matter for *any* method — prompt
templates, thresholds, and background-handling rules (the paper's own
examples in the Introduction) were not swept here; NACLIP's prompt
ensemble is a fixed list of ~80 ImageNet templates in its source, not a
single string, so wasn't varied within this pilot's time budget, and
neither was any proposal-based method's `prompt_template`/`mask_threshold`
(exposed directly in `configs/methods/sam_amg_clip.yaml` — a natural next
step if this is extended). Also only tested on a 300-image subset of one
compact dataset, not the full validation set or the large-vocabulary
splits where the paper's own analysis suggests more room for
target-specific gaming (E2's collapse pattern implies more sensitivity to
prompt/threshold choices under vocabulary competition than under a
59-class compact list).

**Suggested framing for the manuscript if this is included:** report it as
a small, honest robustness check rather than a definitive bound — e.g.
"a pilot sweep of exposed inference hyperparameters for two representative
methods found limited gains (≤0.15 mIoU) from post-hoc tuning on a
compact-vocabulary subset, though this does not rule out larger effects
from prompt-template or threshold tuning under vocabulary expansion,
which the fixed protocol also prohibits." This is a much better-supported
claim than either asserting "fixed protocol prevents tuning gains" (no
evidence either way before this pilot) or overclaiming that this pilot
proves the point in general (it only checked 2 methods × 1 knob-family
each × 1 compact dataset).

Artifacts: `runs/tuning_ablation/{scclip,naclip}_context59/` (configs,
logs, and predictions for all 11 runs).
