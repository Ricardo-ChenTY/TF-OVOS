# Result Comparison and Run Status

Generated: 2026-05-19 04:37 UTC

This file is the quick dashboard for finished numbers, official-alignment status,
and queued runs. All mIoU values are percentages.

---

## Current Queue Order

| Priority | Queue | Status | Purpose |
|----------|-------|--------|---------|
| 1 | `dinov2_sam_after_e2` | running: `dinov2_sam_siglip context59_val` | Proposal+VLM queue; VOC20 is complete. |
| 2 | `official_dense_e2_repair` | running: `NACLIP context459_e2` | Strict official dense large-vocab E2 repair; SCLIP context459 is complete, SCLIP ADE847 will retry after the current queue. |
| 3 | `odise_ctx59_fixed` | running | ODISE ctx59 official eval after fixing the Pascal-Context label-offset issue. |
| 4 | `official_dense_e2_retry_after_repair` | waiting | Re-runs failed dense E2 items with the memory-equivalent synonym aggregation fix. |
| 5 | `trident_cass_after_dinov2` | waiting | Runs Trident/CASS after DINOv2-SAM finishes. |
| 6 | `scclip_after_trident` | waiting | Runs SC-CLIP after Trident/CASS. |
| 7 | `freeda_cliptrase_repair_after_scclip` | waiting | Runs FreeDA/CLIPtrase repair after SC-CLIP. |
| 8 | `official_analysis_watcher` | running | Periodically refreshes analysis outputs. |

GPU is active at 100% utilization. Current active GPU jobs are DINOv2-SAM context59, NACLIP context459 E2, and ODISE ctx59 fixed. Analysis currently sees 48 official records and 385 proposal metrics.

---

## E1 Official Baselines

### Summary

| Method | VOC20 | PC59 | ADE150 | COCO171 | Avg | Official alignment |
|--------|------:|-----:|-------:|--------:|----:|--------------------|
| SCLIP | 81.53* | 34.14 | 16.46 | 22.86 | 38.75 | VOC officialnames now aligned; non-VOC aligned. |
| NACLIP | 83.03* | 38.36 | 19.12 | 25.98 | 41.62 | VOC officialnames fixed run aligned; non-VOC aligned. |
| ResCLIP | 82.31* | 36.85 | 18.07 | 23.59 | 40.20 | No machine-readable official per-dataset table. |
| ProxyCLIP | 80.33* | 39.31 | 19.59 | 26.68 | 41.48 | No machine-readable official per-dataset table. |
| CorrCLIP | 89.01 | 48.68 | 26.96 | 32.22 | 49.22 | Datafix rows aligned; officialnames VOC rerun failed, see notes. |

`*` means VOC20 was re-run with the official VOC class-name file.

### SCLIP - ECCV 2024

| Dataset | Official | Ours | Delta | Status |
|---------|---------:|-----:|------:|--------|
| VOC20 | 81.54 | 81.53 | -0.01 | aligned after officialnames rerun |
| PC59 | 34.46 | 34.14 | -0.32 | aligned |
| ADE150 | 16.45 | 16.46 | +0.01 | aligned |
| COCO171 | 22.77 | 22.86 | +0.09 | aligned |

### NACLIP - WACV 2025

| Dataset | Official | Ours | Delta | Status |
|---------|---------:|-----:|------:|--------|
| VOC20 | 83.03 | 83.03 | +0.00 | aligned after officialnames fixed rerun |
| PC59 | 38.35 | 38.36 | +0.01 | aligned |
| ADE150 | 19.05 | 19.12 | +0.07 | aligned |
| COCO171 | 25.69 | 25.98 | +0.29 | aligned |

### ResCLIP - CVPR 2025

| Dataset | Official | Ours | Status |
|---------|---------:|-----:|--------|
| VOC20 | N/A | 82.31* | officialnames rerun complete |
| PC59 | N/A | 36.85 | no per-dataset official table |
| ADE150 | N/A | 18.07 | no per-dataset official table |
| COCO171 | N/A | 23.59 | no per-dataset official table |

### ProxyCLIP - ECCV 2024

| Dataset | Official | Ours | Status |
|---------|---------:|-----:|--------|
| VOC20 | N/A | 80.33* | officialnames rerun complete |
| PC59 | N/A | 39.31 | no per-dataset official table |
| ADE150 | N/A | 19.59 | no per-dataset official table |
| COCO171 | N/A | 26.68 | no per-dataset official table |

### CorrCLIP - ICCV 2025 Oral

| Dataset | Official | Ours | Delta | Status |
|---------|---------:|-----:|------:|--------|
| VOC20 | 88.8 | 89.01 | +0.21 | aligned on datafix row |
| PC59 | 48.8 | 48.68 | -0.12 | aligned |
| ADE150 | 26.9 | 26.96 | +0.06 | aligned |
| COCO171 | 31.6 | 32.22 | +0.62 | aligned |

CorrCLIP VOC20 officialnames rerun failed because its `eval.py` wrapper does not
accept `--work-dir`. The existing VOC20 datafix row is already aligned with the
official README; retry only if strict officialnames artifacts are needed.

---

## E2 Large-Vocabulary

No official paper baseline is available for these exact E2 splits. Values below
are our measurements.

| Method | Context459 | ADE847 | Status |
|--------|-----------:|-------:|--------|
| ProxyCLIP | 8.41 | 6.90 | complete |
| CorrCLIP | 11.88 | 8.67 | complete |
| MaskCLIP debug adapter | 1.22 | 0.49 | diagnostic only |
| MaskCLIP attention debug | 1.25 | 0.50 | diagnostic only |
| MaskCLIP attention sliding debug | 1.94 | 1.44 | diagnostic only |
| SAM-AMG + CLIP | 4.00 | 5.04 | complete |
| SAM-AMG + SigLIP | 4.97 | 7.24 | complete |

---

## CLIPtrase

TF-OVCOS run, best of available scales.

| Dataset | mIoU | Status |
|---------|-----:|--------|
| VOC20 | 81.20 | complete |
| PC59 | 34.58 | complete |
| ADE150 | 17.04 | complete |
| COCO171 | queued | not yet run |

---

## Proposal-Defined Baselines

These rows are constructed baselines from the proposal, not prior official
methods with external official numbers. The required check is protocol
correctness: same split, official vocabulary where specified, same evaluator,
and no label/index bug.

### E1 Standard Benchmarks

| Method | VOC20 | PC59 | ADE150 | COCO171 | Avg | Status |
|--------|------:|-----:|-------:|--------:|----:|--------|
| SAM-AMG + CLIP | 65.99 | 17.84 | 12.59 | 12.36 | 27.20 | E1 complete |
| SAM-AMG + SigLIP | 65.60 | 19.66 | 15.34 | 14.81 | 28.85 | E1 complete |

### VOC20 Officialnames Rerun

| Method | VOC20 officialnames | Status |
|--------|--------------------:|--------|
| SAM-AMG + CLIP | 62.34 | complete |
| SAM-AMG + SigLIP | 63.00 | complete |

---

## Diffusion / Reference-Based Rows

These rows use official repositories when available. FreeDA E1 is complete.
OVDiff in the current official checkout only implements the VOC dataset key in
`mmdata.py`; non-VOC rows should not be reported as strict official runs unless
we add and validate dataset adapters.

| Method | VOC20 | PC59 | ADE150 | COCO171 | Avg | Status |
|--------|------:|-----:|-------:|--------:|----:|--------|
| FreeDA | 86.19 | 43.42 | 23.19 | 28.80 | 45.40 | E1 complete, official repo/configs |
| OVDiff | 64.56* | - | - | - | - | official VOC complete; non-VOC not implemented in checkout |

`*` OVDiff's printed VOC mIoU is 65.73 including background. The table uses the
proposal VOC20 foreground-class convention, excluding background.

---

## MaskCLIP Rows

The official MaskCLIP repo queue is still pending. The rows below are TF-OVCOS
debug adapters and should not be reported as official MaskCLIP reproduction.

| Method | VOC20 | PC59 | ADE150 | COCO171 | Context459 | ADE847 | Status |
|--------|------:|-----:|-------:|--------:|-----------:|-------:|--------|
| MaskCLIP debug adapter | 37.79 | 8.15 | 1.88 | 4.35 | 1.22 | 0.49 | diagnostic only |
| MaskCLIP attention debug | 37.73 | 8.35 | 1.82 | 4.08 | 1.25 | 0.50 | diagnostic only |
| MaskCLIP attention sliding debug | 49.58 | 11.87 | 4.61 | 7.76 | 1.94 | 1.44 | diagnostic only |
| Official MaskCLIP repo | queued | queued | queued | queued | - | - | waits for VOC officialnames queues |

---

## Notes

### VOC20 Class Names

The original VOC20 runs used `cls_tfovos_voc20.txt`, which standardizes names
such as `boat`, `diningtable`, `person`, and `tvmonitor`. Official repos often
use richer VOC class-name entries. The officialnames rerun uses `cls_voc20.txt`
and separate artifact directories.

SCLIP is now fully aligned on VOC20 after this rerun. NACLIP first improved from
77.98 to 80.60 with the official VOC class-name file; the fixed repair rerun then
matched the official 83.03 exactly.

### COCO Label Mapping

COCO-Stuff171 previously had a label-mapping bug that produced near-zero mIoU.
The canonical rows above use the corrected `datafix` / label-offset behavior.
The current COCO values for SCLIP, NACLIP, CorrCLIP, ProxyCLIP, and ResCLIP are
in the expected range.

### Proposal Queue Safety

The active `proposal_e1_queue` is the restarted, fixed queue. It waits for
`official_voc20_officialnames_queue`, `proposal_voc20_officialnames_queue`, and
`official_maskclip_queue` before running remaining proposal tasks.
