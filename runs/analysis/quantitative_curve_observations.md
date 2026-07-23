# Quantitative Curve Observation Candidates

Generated from current completed analysis CSVs on 2026-05-23.

## Figure Candidates

| Figure | Path | Best use |
| --- | --- | --- |
| Vocabulary collapse curves | `runs/analysis/figures/fig_vocab_collapse_smooth.svg` | Main-text evidence for large-vocabulary class silence. Smooth quadratic fit over measured dataset vocabulary sizes. |
| mIoU vs class-collapse scatter | `runs/analysis/figures/fig_miou_vs_class_collapse_smooth.svg` | Main-text evidence that stronger mIoU does not imply broader class coverage. Smooth trend over large-vocab points. |
| Component-scale curve | `runs/analysis/figures/fig_component_scale_smooth.svg` | Main-text or appendix evidence for small-component failure. Smooth fit over log component area, using dataset-level component bins. |
| Efficiency Pareto scatter | `runs/analysis/figures/fig_efficiency_pareto.svg` | E4 evidence for practical quality-cost tradeoff. |
| Diagnostic decomposition | `runs/analysis/figures/fig_diagnostic_decomposition.svg` | Supporting evidence for localization-vs-naming decomposition. |

Older coarse versions are also kept in the same folder (`fig_vocab_collapse_curves.svg`, `fig_miou_vs_class_collapse.svg`, `fig_component_scale_curve.svg`) for traceability.

## Lightweight Quantitative Claims

### 1. Vocabulary expansion induces class-collapse curves

For SCLIP, NACLIP, ResCLIP, ProxyCLIP, and CorrCLIP:

| Expansion | Mean mIoU loss per +100 classes | Mean zero-IoU class-rate increase per +100 classes |
| --- | ---: | ---: |
| Context-59 -> Context-459 | 7.74 mIoU | +10.2 pp |
| ADE-150 -> ADE-847 | 1.91 mIoU | +5.4 pp |

Strongest instance: CorrCLIP has the highest large-vocabulary mIoU among these rows, but also the largest collapse: Context-459 zero-IoU rate 44.2% and ADE-847 zero-IoU rate 49.9%.

### 2. Higher large-vocabulary mIoU can coexist with higher collapse

Across the ten large-vocabulary points (5 methods x Context-459/ADE-847):

| Relationship | Value |
| --- | ---: |
| Pearson correlation between mIoU and zero-IoU class rate | 0.597 |
| Spearman correlation between mIoU and zero-IoU class rate | 0.721 |

Dataset-wise correlations:

| Dataset | Pearson | Spearman |
| --- | ---: | ---: |
| Context-459 | 0.858 | 0.600 |
| ADE-847 | 0.960 | 0.700 |

This supports the observation that top-line mIoU gains are concentrated rather than uniformly improving class coverage.

### 3. Small components collapse across methods

Weighted mean component IoU across completed datasets:

| Method | Large | Medium | Small | Small / large |
| --- | ---: | ---: | ---: | ---: |
| SCLIP | 0.2571 | 0.0631 | 0.0124 | 0.048 |
| NACLIP | 0.3034 | 0.0905 | 0.0175 | 0.058 |
| ResCLIP | 0.2827 | 0.0775 | 0.0151 | 0.053 |
| ProxyCLIP | 0.2980 | 0.0784 | 0.0149 | 0.050 |
| CorrCLIP | 0.3699 | 0.1133 | 0.0184 | 0.050 |

Mean small/large ratio is 0.052, i.e. roughly a 94.8% drop from large to small components.

### 4. Efficiency has a small Pareto frontier

Among strict training-free rows with timing available, the current Pareto frontier is:

| Method | E1 avg mIoU | sec/img |
| --- | ---: | ---: |
| MaskCLIP-Attn | 13.00 | 0.030 |
| Trident | 43.31 | 0.080 |
| CorrCLIP | 49.22 | 0.193 |

This supports an E4 observation that several slower methods are not Pareto-leading under the current fixed protocol.

