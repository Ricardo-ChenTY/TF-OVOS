# Table 11 E2 Mismatch Summary

| method_group | localized_pairs | mismatch_pairs | MCMR@0.5_pooled | MCMR@0.5_mean_of_cells | GT_region_top1_proxy | proposal_recall_at_0.5_proxy | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense-map TF methods | 81655 | 28039 | 0.3434 | 0.3659 | 0.2973 | 0.2798 | computed_proxy_from_dense_label_maps |
| Detector+SAM TF methods |  |  |  |  |  |  | artifact_gap |
| Proposal+naming TF methods |  |  |  |  |  |  | artifact_gap |
| CLIP + VFM TF methods | 96002 | 37533 | 0.3910 | 0.4070 | 0.3277 | 0.4164 | computed_proxy_from_dense_label_maps |
| Diffusion/reference TF methods | 21811 | 9937 | 0.4556 | 0.4764 | 0.2990 | 0.3785 | computed_proxy_from_dense_label_maps |
| MCC consistency diagnostics |  |  |  |  |  |  | artifact_gap |
| Trained references | 30060 | 9086 | 0.3023 | 0.3181 | 0.4213 | 0.5227 | computed_proxy_from_dense_label_maps |
