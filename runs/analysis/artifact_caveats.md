# Artifact and Metric Caveats

## 8-bit large-vocabulary prediction maps

`official_prediction_map_metrics_biou.csv` flags **16** method-dataset rows with `possible_8bit_large_vocab_artifact=True`. These are ADE-847 and Context-459 rows whose saved prediction PNGs cap labels at 255, while the benchmark vocabularies contain more classes.

Implication: use large-vocabulary artifact-derived metrics such as semantic BIoU, predicted-class coverage, and confusion pairs as diagnostics, not as the sole basis for main leaderboard claims. Official log mIoU/class tables remain the primary source for E1/E2 numeric results.

Affected rows:
- cass / ade847: max raw label 255, predicted class coverage 0.302
- cass / context459: max raw label 255, predicted class coverage 0.556
- corrclip / ade847: max raw label 255, predicted class coverage 0.301
- corrclip / context459: max raw label 255, predicted class coverage 0.551
- naclip / ade847: max raw label 255, predicted class coverage 0.302
- naclip / context459: max raw label 255, predicted class coverage 0.556
- proxyclip / ade847: max raw label 255, predicted class coverage 0.302
- proxyclip / context459: max raw label 255, predicted class coverage 0.558
- resclip / ade847: max raw label 255, predicted class coverage 0.302
- resclip / context459: max raw label 255, predicted class coverage 0.558
- scclip / ade847: max raw label 255, predicted class coverage 0.302
- scclip / context459: max raw label 255, predicted class coverage 0.556
- sclip / ade847: max raw label 255, predicted class coverage 0.302
- sclip / context459: max raw label 255, predicted class coverage 0.558
- trident / ade847: max raw label 255, predicted class coverage 0.302
- trident / context459: max raw label 255, predicted class coverage 0.553

## Diagnostic proxy status

MCMR, proposal recall, and GT-text localization rows are computed from saved dense label maps as proxy diagnostics unless a method exposes true proposal masks/region scores. Phrase as failure-mode probes rather than official benchmark metrics.
