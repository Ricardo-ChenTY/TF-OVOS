# Table 9 Diagnostic Probes

| method | dataset | gt_region_naming_top1 | gt_text_localization_iou | gt_text_localization_biou | proposal_oracle_iou | proposal_recall_at_05 | mcmr_at_05 | diagnostic_source | artifact_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| san | context459 | 0.5139 | 0.4082 |  | 0.5845 | 0.6302 | 0.2770 | proxy_from_dense_label_map | Top-5 naming, true proposal recall, and MCC before/after need saved region scores/proposal masks/MCC rerank artifacts. |
| san | ade847 | 0.3288 | 0.2403 |  | 0.4064 | 0.4153 | 0.3591 | proxy_from_dense_label_map | Top-5 naming, true proposal recall, and MCC before/after need saved region scores/proposal masks/MCC rerank artifacts. |
