# Novel Observation Candidates

This note separates observations that look genuinely under-discussed in existing OVSS papers from supporting or qualitative findings. The strongest claims are framed conservatively: prior work has discussed related issues such as background absorption, embedding bias, semantic duplication, and long-tail large-vocabulary difficulty, but usually reports aggregate mIoU rather than explicitly quantifying class-level silence or coverage collapse.

## A. Strongest Under-Discussed Quantitative Findings

### 1. Large-vocabulary class silence / zero-IoU collapse

**Claim.** Strong OVSS methods improve average mIoU while leaving a large fraction of fine-grained vocabulary classes completely silent.

**Why this seems under-discussed.** Existing OVSS tables usually report mIoU on A-847 / PC-459, but do not explicitly report the fraction of classes with zero IoU. Related work discusses background collapse, known-category collapse, semantic duplication, or long-tail difficulty, but not this exact coverage-collapse rate as a main diagnostic.

**Current evidence.**

| Method | Context-459 zero-IoU | ADE-847 zero-IoU |
| --- | ---: | ---: |
| SCLIP | 39.7% | 39.2% |
| NACLIP | 38.6% | 37.2% |
| ResCLIP | 39.4% | 39.1% |
| ProxyCLIP | 42.4% | 42.0% |
| CorrCLIP | 44.2% | 49.9% |

**Light quantification.**

| Expansion | Mean mIoU loss per +100 classes | Mean zero-IoU rate increase per +100 classes |
| --- | ---: | ---: |
| Context-59 -> Context-459 | 7.74 mIoU | +10.2 pp |
| ADE-150 -> ADE-847 | 1.91 mIoU | +5.4 pp |

**Figure.** `runs/analysis/figures/fig_vocab_collapse_curves.svg`

**Paper wording.** Higher large-vocabulary mIoU does not necessarily mean broader vocabulary coverage; it may reflect better performance on a subset of classes while many fine-grained labels remain unpredicted.

### 2. Accuracy-coverage mismatch: better mIoU can coexist with worse collapse

**Claim.** Large-vocabulary mIoU and class coverage are not the same axis of progress.

**Current evidence.** CorrCLIP is the strongest strict TF row by E1 average mIoU and large-vocabulary mIoU, but it also has the highest zero-IoU collapse rate among the current strong-method set: 44.2% on Context-459 and 49.9% on ADE-847.

**Light quantification.**

Across 10 large-vocabulary points (5 methods x Context-459/ADE-847), mIoU and zero-IoU class rate have Pearson correlation 0.597 and Spearman correlation 0.721. Dataset-wise Pearson is 0.858 on Context-459 and 0.960 on ADE-847.

**Figure.** `runs/analysis/figures/fig_miou_vs_class_collapse.svg`

**Paper wording.** The best average method is not necessarily the best coverage method; leaderboard gains can be concentrated rather than uniformly distributed across the vocabulary.

### 3. Small-component degradation is nearly universal

**Claim.** Current OVSS methods lose small connected components almost completely, even when large regions are segmented reasonably.

**Current evidence.**

| Method | Large component IoU | Medium component IoU | Small component IoU | Small / large |
| --- | ---: | ---: | ---: | ---: |
| SCLIP | 0.2571 | 0.0631 | 0.0124 | 0.048 |
| NACLIP | 0.3034 | 0.0905 | 0.0175 | 0.058 |
| ResCLIP | 0.2827 | 0.0775 | 0.0151 | 0.053 |
| ProxyCLIP | 0.2980 | 0.0784 | 0.0149 | 0.050 |
| CorrCLIP | 0.3699 | 0.1133 | 0.0184 | 0.050 |

**Light quantification.** Mean small/large ratio is 0.052, roughly a 94.8% drop from large to small components.

**Figure.** `runs/analysis/figures/fig_component_scale_curve.svg`

**Paper wording.** The problem is not only semantic naming; spatial granularity collapses at the component level.

### 4. Proposal quality and naming quality trade off

**Claim.** Proposal+naming pipelines can have better localization ceilings but worse category assignment than dense-map methods.

**Current evidence.**

| Method family | Proposal Recall@0.5 proxy | MCMR@0.5 proxy |
| --- | ---: | ---: |
| Dense-map TF methods | 0.416 | 0.297 |
| Proposal+naming TF methods | 0.557 | 0.345 |

**Interpretation.** Proposal methods find plausible regions more often, but naming those regions remains weaker.

**Paper wording.** Better masks do not automatically solve open-vocabulary semantic assignment.

### 5. Practical ranking is a Pareto problem, not an mIoU-only problem

**Claim.** Some methods that look competitive by mIoU are not competitive once speed is included.

**Current Pareto front among strict TF rows with timing.**

| Method | E1 avg mIoU | sec/img |
| --- | ---: | ---: |
| MaskCLIP-Attn | 13.00 | 0.030 |
| Trident | 43.31 | 0.080 |
| CorrCLIP | 49.22 | 0.193 |

**Figure.** `runs/analysis/figures/fig_efficiency_pareto.svg`

**Paper wording.** E4 changes the practical interpretation of E1: several high-cost rows are dominated by faster and stronger alternatives under the fixed protocol.

## B. Medium-Strength Quantitative / Diagnostic Findings

### 6. Naming and localization errors are separable

**Claim.** Errors do not come from one source; naming, text-conditioned localization, and proposal recall can be separated with diagnostic probes.

**Current CorrCLIP averages.**

| Probe | Value |
| --- | ---: |
| GT-region naming Top-1 | 0.487 |
| GT-text localization IoU | 0.403 |
| Proposal-oracle IoU | 0.605 |
| Proposal Recall@0.5 | 0.635 |
| MCMR@0.5 | 0.352 |

**Figure.** `runs/analysis/figures/fig_diagnostic_decomposition.svg`

**Paper wording.** The same method can have a reasonable proposal/localization ceiling while still suffering category mismatch.

### 7. Large vocabulary stress is not identical across datasets

**Claim.** Context expansion and ADE expansion produce different collapse slopes.

**Current evidence.** Context-59 -> Context-459 has larger mean mIoU loss per +100 classes than ADE-150 -> ADE-847 (7.74 vs. 1.91), while both produce strong zero-IoU collapse increases.

**Interpretation.** Vocabulary size alone is not sufficient; taxonomy structure and class distribution matter.

### 8. Stronger CLIP/VFM methods are not uniformly better across failure modes

**Claim.** CorrCLIP improves average mIoU and component IoU, but does not solve class silence and may worsen it under large vocabularies.

**Interpretation.** This supports a multi-axis benchmark story: mIoU, coverage, component scale, and mismatch rate reveal different method behaviors.

## C. Qualitative / Semi-Quantitative Findings

### 9. Mismatch pairs are semantically structured, not random noise

**Claim.** Many errors form reusable confusion communities, such as material/texture categories, structure categories, and object-stuff substitutions.

**Evidence source.** `runs/analysis/table11_e2_mismatch_pairs_biou.csv`

**Use.** Best as qualitative examples beside the MCMR table.

### 10. Large-vocab failure is partly taxonomy granularity failure

**Claim.** Some collapse and mismatch comes from fine-grained label competition, not merely inability to segment the object.

**Evidence source.** MCMR pairs, zero-IoU classes, and large-vocabulary drop.

**Use.** Phrase conservatively unless synonym-collapsed mIoU or hierarchy-aware IoU is fully computed.

### 11. Proposal pipelines are explainable but cost-heavy

**Claim.** SAM/DINO proposal pipelines are easier to decompose into proposal recall and naming, but current fixed-protocol versions are slower and lower mIoU.

**Evidence source.** E1, E4, and proposal/naming diagnostic rows.

## D. Best Main-Paper Package

Recommended main observation section:

1. **Class silence under vocabulary expansion.** Use `fig_vocab_collapse_curves.svg`.
2. **mIoU-coverage mismatch.** Use `fig_miou_vs_class_collapse.svg`.
3. **Small-component degradation.** Use `fig_component_scale_curve.svg`.
4. **Mask-good/name-hard decomposition.** Use family-level MCMR/proposal recall table.

Recommended appendix/supporting:

1. Efficiency Pareto scatter.
2. Diagnostic decomposition bar chart.
3. Top mismatch-pair examples.
4. Full per-method collapse table.

