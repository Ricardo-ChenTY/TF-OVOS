# Adapter Plan

Each method gets a thin adapter that hides its native repository and emits the
common prediction format.

## Dense CLIP / OVSS Maps

Methods: MaskCLIP, CLIP-DIY, SCLIP, NACLIP, ProxyCLIP, CorrCLIP, Trident.

Adapter steps:

1. Run the native method with the fixed vocabulary.
2. Convert class score maps to per-class candidate binary masks.
3. Keep connected components above `min_component_area`.
4. Score candidates by class confidence and optional mask quality.
5. Emit the best `(mask, label)` pair.

## GroundingDINO + SAM/SAM2

Adapter steps:

1. Prompt GroundingDINO with all class names using fixed templates.
2. Convert detector boxes to SAM/SAM2 masks.
3. Score each mask-category pair by detector score and mask stability.
4. Emit the best pair.

This is the fastest first baseline because the pair structure already matches
the benchmark output.

## Proposal + VLM Naming

Methods: SAM-AMG + CLIP/SigLIP, DINOv2 + SAM + CLIP/SigLIP, MCC variants.

Adapter steps:

1. Generate class-agnostic proposals.
2. Crop or alpha-mask each proposal.
3. Score proposal crops against the fixed vocabulary.
4. Optionally apply MCC reranking.
5. Emit the best pair.

## Diffusion / Reference Rows

Methods: FreeDA, OVDiff.

Treat these as later-stage integrations. They have heavier offline artifacts and
older dependency stacks, so keep them isolated from the core benchmark package.
