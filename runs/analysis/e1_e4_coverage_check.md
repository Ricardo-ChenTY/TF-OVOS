# E1-E4 Coverage Check

Policy: every method with E1 results should be evaluated under the shared protocol for E1/E2, then included in E3/E4 analysis where the metric is applicable.

## Already complete for E1+E2 standard mIoU

- TF-OVCOS adapters: maskclip, maskclip_attn, maskclip_attn_slide, sam_amg_clip, sam_amg_siglip
- Official/repo methods: ProxyCLIP, CorrCLIP

## Queued to fill E2 gaps

- official_dense_e2_after_refs: SCLIP, NACLIP, ResCLIP on Context459/ADE847
- dinov2_sam_after_e2: DINOv2+SAM+SigLIP and DINOv2+SAM+CLIP on all E1+E2 datasets
- trident_cass_after_dinov2: Trident and CASS on all E1+E2 datasets
- scclip_after_trident: SC-CLIP on all E1+E2 datasets
- freeda_cliptrase_repair_after_scclip: CLIPtrase COCO171/Context459/ADE847 plus FreeDA Context459/ADE847

## E1 complete but E2 still needs adapter/config work

- FreeDA: E1 complete; Context459/ADE847 old-mmseg dataset registry/config smoke passed and is now queued after SC-CLIP.
- CLIPtrase: VOC20/Context59/ADE150 complete; COCO171/Context459/ADE847 path/suffix smoke passed and is now queued after SC-CLIP.
- OVDiff: only VOC works in current checkout; other datasets require implementing official-style mmdata dataset adapters.

## E3/E4 analysis status

- E3 rank/worst-target summaries are computed from completed E1 targets and will automatically densify as queued methods finish.
- E4 runtime/cost rows are partially computed from logs/runtime.json; proposal-specific E4 probes require proposal artifacts and are only applicable to SAM/Detection/SAM-DINO families.
