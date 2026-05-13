# TF-OVOS Benchmark — Execution Plan

Goal: fill in the complete E1–E4 tables for the paper.

---

## Target Tables

### E1 — Effectiveness (mIoU, standard tier)

| Method | Family | VOC20 | Ctx-59 | ADE-150 | COCO-171 | **mean** |
|--------|--------|-------|--------|---------|----------|----------|
| MaskCLIP-lite | clip_dense | 37.8 | 8.1 | 1.9 | 0.2 | — |
| **MaskCLIP** | clip_dense | | | | | |
| NACLIP | clip_dense | | | | | |
| SCLIP | clip_dense | | | | | |
| CorrCLIP | clip_vfm | | | | | |
| ResClip | clip_dense | | | | | |
| CLIP-DIY | clip_dense | | | | | |
| CLIPtrase | clip_dense | | | | | |
| CASS | clip_vfm | | | | | |
| FreeDA | diffusion | | | | | |
| OVDiff | diffusion | | | | | |
| Trident | clip_vfm | | | | | |
| DINOv2+SAM+CLIP | proposal | | | | | |
| DINOv2+SAM+SigLIP | proposal | | | | | |
| SAM-AMG+SigLIP | proposal | | | | | |
| GroundingDINO+SAM2 | detector_sam | | | | | |
| *OVSeg† | trained_ref | | | | | |
| *SAN† | trained_ref | | | | | |
| *ODISE† | trained_ref | | | | | |

† trained reference, not strict TF — included for comparison only.

---

### E2 — Vocabulary Robustness

| Method | Ctx-59 | Ctx-459 | **Δ_ctx** | ADE-150 | ADE-847 | **Δ_ade** | **Δ_vocab** | MCMR@0.5 |
|--------|--------|---------|-----------|---------|---------|-----------|-------------|----------|
| MaskCLIP-lite | 8.1 | | | 1.9 | | | | 0.0 |
| MaskCLIP | | | | | | | | |
| NACLIP | | | | | | | | |
| … (all methods) | | | | | | | | |

Δ_vocab = avg(Δ_ctx + Δ_ade).  Lower is better (robust to larger vocab).

---

### E3 — Cross-Dataset Generalization

Trained on: VOC20 + Ctx-59 + ADE-150 + COCO-171 (E1 splits).
Tested on: unseen subsets / held-out splits.

| Method | VOC20→Ctx | Ctx→ADE | ADE→COCO | **avg-worst** |
|--------|-----------|---------|----------|---------------|
| MaskCLIP-lite | | | | |
| … | | | | |

*(Exact transfer protocol TBD once E1 numbers are in.)*

---

### E4 — Efficiency

Measured automatically via `runtime.json` written by `run_method.py`.

| Method | img/s (V100) | GPU mem (GB) | FLOPs (G) |
|--------|-------------|--------------|-----------|
| MaskCLIP-lite | | | |
| … | | | |

---

## Implementation Roadmap

### Phase 1 — CLIP-dense family (all share ViT-B/16 backbone, ~1 day each)

These methods differ only in *how* patch tokens are processed before cosine similarity.
They can reuse `MaskClipAdapter._encode_text` and `_preprocess`.

1. **MaskCLIP (attention surgery)** — `adapters/maskclip.py` upgrade
   - In `_encode_patch_tokens`: for the last N layers, replace key-query attention
     with key-key self-similarity (MaskCLIP §3.2).
   - Expected: VOC20 ~50%, ADE150 ~15%.
   - Paper: Zhou et al., ECCV 2022.

2. **NACLIP** — new `adapters/naclip.py`
   - After getting patch tokens, smooth each token with its spatial neighbours
     (3×3 or 5×5 Gaussian-weighted sum) before cosine similarity.
   - Expected: ADE150 ~17%, Ctx59 ~18%.
   - Paper: Hajimiri et al., ECCV 2024.

3. **SCLIP** — new `adapters/sclip.py`
   - Replace standard self-attention in last layer with *correlative* self-attention:
     Q←value, K←value, V←value (removes position bias).
   - Expected: ADE150 ~18%.
   - Paper: Wang et al., ECCV 2024.

4. **ResClip** — new `adapters/resclip.py`
   - Add residual connections from early ViT layers to late layers before
     computing dense similarity (multi-scale aggregation).
   - Paper: Chen et al., NeurIPS 2024.

5. **CLIP-DIY** — new `adapters/clip_diy.py`
   - Patch-level CLIP + GrabCut-style iterative refinement using attention maps.
   - Paper: Wysoczańska et al., WACV 2024.

6. **CLIPtrase** — new `adapters/cliptrase.py`
   - Use GradCAM traces on CLIP text tokens to generate dense maps.
   - Paper: Shao et al., ECCV 2024.

---

### Phase 2 — VFM-augmented CLIP (DINO/SAM backbone, ~2 days each)

7. **CorrCLIP** — new `adapters/corrclip.py`
   - Compute patch-patch correlation from DINOv2 features, reweight CLIP similarity.
   - Deps: `dinov2` (Facebook), `open_clip`.
   - Paper: Sun et al., 2024.

8. **CASS** — new `adapters/cass.py`
   - Spectral clustering on DINOv2 patch features → segment proposals → CLIP naming.
   - Deps: `dinov2`, `open_clip`, `scipy` (for eigendecomposition).
   - Paper: CASS, 2024.

9. **Trident** — new `adapters/trident.py`
   - CLIP + DINOv2 + SAM multi-scale feature fusion, training-free.
   - Deps: `dinov2`, `segment_anything`, `open_clip`.
   - Paper: Trident, 2024.

---

### Phase 3 — Proposal-based (SAM/detector generates masks, VLM names them, ~2 days each)

10. **DINOv2 + SAM + CLIP** — `adapters/dinov2_sam_clip.py`
    - SAM AMG generates ~100 masks per image.
    - For each mask, crop + CLIP encode → argmax over vocab.
    - Merge by majority vote into semantic label map.
    - Deps: `segment_anything`, `dinov2`, `open_clip`.

11. **DINOv2 + SAM + SigLIP** — `adapters/dinov2_sam_siglip.py`
    - Same as above but swap CLIP for SigLIP (better zero-shot naming).
    - Deps: `transformers` (HF SigLIP).

12. **SAM-AMG + SigLIP** — `adapters/sam_amg_siglip.py`
    - Pure SAM AMG (no DINOv2 guidance) + SigLIP naming.
    - Simpler variant; tests SAM mask quality independently.

13. **GroundingDINO + SAM2** — `adapters/groundingdino_sam2.py`
    - For each vocab class, run GroundingDINO to get bounding boxes.
    - Feed boxes into SAM2 to get fine masks.
    - Merge all class masks into semantic label map.
    - Deps: `groundingdino`, `sam2`.

---

### Phase 4 — Diffusion-based (slower, run last)

14. **FreeDA** — `adapters/freeda.py`
    - Offline: extract Stable Diffusion cross-attention maps per class.
    - Online: match to test image via prototype matching.
    - Deps: `diffusers`, `transformers`.

15. **OVDiff** — `adapters/ovdiff.py`
    - Use diffusion inpainting score as class likelihood at each pixel.
    - Deps: `diffusers`.

---

### Phase 5 — Trained references (run from official repos or use published numbers)

16. **OVSeg** — use official inference code + published weights.
17. **SAN** — use official inference code + published weights.
18. **ODISE** — use official inference code + published weights.

These are not strict TF methods; they appear in a separate section of Table 1
as upper-bound references.

---

## Dataset Coverage per Phase

| Phase | E1 datasets | E2 datasets | Appendix |
|-------|------------|-------------|----------|
| After Phase 1 | voc20, ctx59, ade150, coco171 | ctx459, ade847 | ovcamo, camo, cod10k, nc4k |
| After Phase 2 | same | same | same |
| After Phase 3 | same | same | same |
| After Phase 4 | same | same | same |

Run order per method:
```bash
python -m tf_ovos.run_benchmark --method <name> \
    --dataset voc20_val --dataset context59_val \
    --dataset ade20k150_val --dataset coco_stuff171_val \
    --dataset context459_val --dataset ade20k847_val \
    --num-shards 4
```

---

## Dependencies to Install (cumulative)

```bash
# Phase 1 (already done)
pip install open-clip-torch scipy

# Phase 2
pip install git+https://github.com/facebookresearch/dinov2.git

# Phase 3
pip install segment-anything
pip install git+https://github.com/facebookresearch/sam2.git
pip install groundingdino-py
pip install transformers  # for SigLIP

# Phase 4
pip install diffusers accelerate
```

---

## Current Status

- [x] Benchmark harness (E1–E4 pipeline, metrics, manifests)
- [x] All 6 datasets downloaded and manifests generated
- [x] MaskCLIP-lite: E1 complete (VOC20=37.8, Ctx59=8.1, ADE150=1.9, COCO171=0.2)
- [ ] MaskCLIP (surgery): E1 + E2
- [ ] NACLIP, SCLIP, ResClip, CLIP-DIY, CLIPtrase: E1 + E2
- [ ] CorrCLIP, CASS, Trident: E1 + E2
- [ ] DINOv2+SAM+CLIP/SigLIP, SAM-AMG+SigLIP, GDino+SAM2: E1 + E2
- [ ] FreeDA, OVDiff: E1 + E2
- [ ] Appendix datasets: all methods
- [ ] E3 cross-dataset analysis
- [ ] E4 efficiency table (auto-collected, just needs aggregation)
- [ ] summarize_results → CSV → paper tables
