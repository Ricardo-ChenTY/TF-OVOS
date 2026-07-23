# Extra Family Preparation

Updated: 2026-05-18 22:10 UTC

## Added / prepared

| Family | Method | Repo | Current status | Next action |
| --- | --- | --- | --- | --- |
| self-calibrated CLIP | SC-CLIP | `third_party/official_methods/SC-CLIP` | repo cloned; current `tf-ovos` env import smoke passed; TF-OVOS E1/E2 configs generated for VOC20, Context59, ADE150, COCO-Stuff171, Context459, ADE847 | queued in `tmux:scclip_after_trident`, waiting behind ODISE -> OVSeg -> dense E2 -> DINOv2-SAM -> Trident/CASS |
| label / pixel propagation | LPOSS | `third_party/official_methods/LPOSS` | repo cloned; official code needs old `mmcv-full<1.7` / `mmsegmentation==0.27`; current env import stops at version guard | prepare separate legacy env or port registry imports |
| prompt-template / data-centric | FLOSS | `third_party/official_methods/FLOSS` | repo cloned; pretrained template ranking JSONs are present for MaskCLIP/NACLIP/CLIP-DINOiser on VOC20, Context59, ADE150, COCO-Stuff | likely add as prompt-template augmentation rows after current queues, not as a pure standalone model |
| data-centric reference | ReME | `third_party/official_methods/ReME` | repo cloned; current public repo contains README + dataset prepare scripts, but no direct eval/model code in checkout | keep as literature/method-family reference unless executable code or reference index is released |

## Current GPU queue

The active queue order is unchanged for already-running work:

1. `odise_partial_fixed` currently running.
2. `ovseg_context59_fix_after_odise` waits for ODISE.
3. `official_dense_e2_after_refs` waits for refs, then fills SCLIP/NACLIP/ResCLIP E2.
4. `dinov2_sam_after_e2` waits, then runs DINOv2-SAM variants E1/E2.
5. `trident_cass_after_dinov2` waits, then runs Trident/CASS E1/E2.
6. `scclip_after_trident` waits, then runs SC-CLIP E1/E2.

SC-CLIP is queued after Trident/CASS so it will not steal GPU from the current main sequence.
