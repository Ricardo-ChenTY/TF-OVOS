# Bootstrap Robustness Checks

| Metric | Estimate | 95% CI | Units |
|---|---:|---:|---:|
| large_vocab_zero_iou_rate_pp | 41.162 | [39.278, 43.490] | 10 |
| vocab_expansion_miou_loss | 22.134 | [16.365, 27.753] | 10 |
| vocab_expansion_zero_iou_increase_pp | 39.096 | [36.544, 41.749] | 10 |
| miou_zero_iou_pearson | 0.597 | [0.296, 0.927] | 10 |
| miou_zero_iou_spearman | 0.721 | [0.091, 0.987] | 10 |
| small_large_component_iou_ratio | 0.061 | [0.053, 0.069] | 30 |
| small_component_relative_drop_pp | 93.930 | [93.049, 94.727] | 30 |
| proposal_recall_minus_mcmr_pp | 15.565 | [3.380, 29.151] | 30 |
| localization_iou_minus_naming_top1_pp (negative = naming higher) | -10.961 | [-12.220, -9.677] | 30 |

Notes: intervals are resampled over method-dataset rows or paired compact/large comparisons from existing outputs. They are intended as quick robustness checks for paper wording.
