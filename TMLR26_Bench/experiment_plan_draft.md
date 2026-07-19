# TMLR26 Fix/Experiment Plan (Draft, pre-repo-audit)

Drafted from a manual review of the submitted PDF only, before checking the
repository for existing source data. Superseded by `plan.md` in this
directory once the repo audit confirmed which issues are code bugs vs
missing experiments. Kept here for traceability.

## Tier 0: core, must fix before resubmission

| # | Experiment | Issue it fixes | Method scope | Datasets | Output | Effort |
|---|------------|-----------------|---------------|----------|--------|--------|
| E-1 | Per-target tuning delta (fixed recipe vs. tuned) | Core motivation (§3.1 training-free rule) has no supporting experiment | NACLIP, CorrCLIP, FreeDA, SAM-AMG+SigLIP | Ctx-59, ADE-150, COCO-Stuff (+ Ctx-459 optional) | New table: fixed vs. per-target-tuned mIoU, Δ | Medium: 3-5 hyperparameter sweeps/method/dataset |
| E-2 | Table 9 MCMR recompute + bug audit | Table 9 family aggregate contradicts Table 2 per-method MCMR and reverses the dense-vs-proposal ordering | All dense + all proposal methods | Ctx-459, ADE-847 | Corrected family-level MCMR aggregation | Small: aggregation-only, no new inference (pending root-cause check) |
| E-3 | ZIoU x mIoU correlation + CLIPtrase anomaly check | Obs 4 claim ("ZIoU tracks the collapse") contradicted by CLIPtrase/SC-CLIP | All 18 methods (existing numbers) | Ctx-459, ADE-847 | Spearman ρ, per-class IoU histogram for CLIPtrase | Small: stats only |

## Tier 1: structural, strongly recommended

| # | Experiment | Issue | Scope | Datasets | Output | Effort |
|---|------------|-------|-------|----------|--------|--------|
| E-4 | Full E4 cost table (memory, #calls, T_offline, T_amortized) | E4 has no table, FreeDA (the only offline-stage method) is missing from Fig. 4 | All Table 1 methods, FreeDA mandatory | fixed N images | New table | Medium: needs profiling instrumentation |
| E-5 | Expand 4-probe diagnostic beyond CorrCLIP | Obs 5 rests on one method's numbers | NACLIP, SC-CLIP, Trident, CASS, SAM-AMG+SigLIP, FreeDA | Ctx-459, ADE-847 | Probe table, 4 metrics x 6 methods | Medium-large: needs GT-conditioned inference path |
| E-6 | Component-size diagnostic expansion + compact/large split | Table 8 only has 5 methods but Obs 8/9 claim "universal"/"holds across... methods" | All Table 1 methods | split compact vs. large | Two new cIoU tables | Medium: reuses existing connected-component code |
| E-7 | Reproduction-gap table (paper-reported vs. reproduced) | No explanation for why e.g. SCLIP/NACLIP reproduced numbers are below original papers | Methods with a citable original number | VOC-20/Ctx-59/ADE-150/COCO | New table + delta source notes | Small: literature lookup, no new inference |

## Tier 2: time-permitting

| # | Experiment | Issue | Notes |
|---|------------|-------|-------|
| E-8 | Resolve §3.2-promised-but-unrun methods (OVSeg/ODISE/CAT-Seg/FC-CLIP/OVDiff/GroundingDINO+SAM) | Listed in text/tables but never populated | Either run or cut references |
| E-9 | Real cross-domain E3 evidence (Cityscapes or prompt-bank swap) | E3 is currently `min(E1)` in disguise, and Obs 6 admits it | Larger addition |
| E-10 | Publish actual run-card config values | "Fixed protocol" claimed but no actual resolution/prompt/NMS values shown | Documentation only |

---

**Status note (superseded):** this draft assumed all of the above required
new GPU inference. The repo audit (see `plan.md`) found that E-2, E-3, and
part of E-6/E-7 already have real source CSVs sitting in
`tf_ovcos_full_results_20260526/runs/analysis/`, and that Table 9's bug has
a concrete one-line root cause in `scripts/generate_diagnostic_tables.py`.
See `plan.md` for the corrected, repo-grounded plan.
