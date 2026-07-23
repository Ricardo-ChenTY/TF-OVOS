# Table 9 Diagnostic Probes

| method | dataset | gt_region_naming_top1 | gt_text_localization_iou | gt_text_localization_biou | proposal_oracle_iou | proposal_recall_at_05 | mcmr_at_05 | diagnostic_source | artifact_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sclip | voc20 | 0.8966 | 0.6936 | 0.5791 | 0.7159 | 0.7241 | 0.0476 | proxy_from_dense_label_map | Top-5 naming, true proposal recall, and MCC before/after need saved region scores/proposal masks/MCC rerank artifacts. |
