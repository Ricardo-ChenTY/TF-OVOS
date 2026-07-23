# Four-Event Evidence Package

This file summarizes paper-ready evidence from existing runs. Bootstrap intervals resample the available method–dataset or paired units, so they are lightweight robustness checks rather than a replacement for per-image statistical testing.

## Event 1: Large-Vocabulary Class Silence

Mean zero-IoU class rate across strong methods on Context-459/ADE-847: **41.162 pp** (95% bootstrap CI 39.278, 43.490).
Paired compact→large expansion increases zero-IoU rate by **39.096 pp** (CI 36.544, 41.749) and loses **22.134 mIoU** (CI 16.365, 27.753).

Use with: `fig_vocab_collapse_curves.svg`, `fig_miou_vs_class_collapse.svg`, and `official_best_metrics.csv`.

## Event 2: mIoU-Coverage Mismatch

Across large-vocabulary points, mIoU and zero-IoU collapse correlate positively rather than negatively: Pearson **0.597** (CI 0.296, 0.927); Spearman **0.721** (CI 0.091, 0.987).
Interpretation: leaderboard gains can concentrate on classes that remain predictable while many fine-grained classes stay silent.

## Event 3: Small-Component Degradation

Small/large component IoU ratio: **0.061** (CI 0.053, 0.069).
Relative drop from large to small components: **93.930%** (CI 93.049, 94.727).

Use with: `fig_component_scale_curve.svg`, `fig_component_scale_smooth.svg`, and `exploratory_scale_components.csv`.

## Event 4: Localization vs. Naming / Semantic Binding

Proposal/localization recall exceeds MCMR proxy by **15.565 pp** (CI 3.380, 29.151).
GT-region naming top-1 and GT-text localization IoU differ by **10.961 pp** on average in the naming-favored direction (localization minus naming CI -12.220, -9.677), reinforcing that naming and localization are separable axes rather than one scalar failure mode.

Use with: `fig_diagnostic_decomposition.svg`, `table9_diagnostic_probes_biou.csv`, and `case_study_mismatch_signboard.*`.

## BioU / Boundary Evidence

- voc20: mean semantic BIoU 0.5092
- context59: mean semantic BIoU 0.0707
- ade20k: mean semantic BIoU 0.0516
- coco_stuff164k: mean semantic BIoU 0.0329
- context459: mean semantic BIoU 0.0164
- ade847: mean semantic BIoU 0.0119

Boundary quality drops sharply outside compact VOC-style settings, but ADE-847/Context-459 rows must be phrased as diagnostic because several saved official prediction maps are 8-bit label images.
