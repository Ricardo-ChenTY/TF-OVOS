# E1-E4 Main Summary

Scope: this summary covers the main E1-E4 evidence only. Appendix/Table10 hard-domain results are intentionally excluded until that run finishes and is summarized separately.

## Takeaways

- E1: proposal+naming variants dominate the local TF-OVCOS adapter set. `sam_amg_siglip` is the current best average on the four standard semantic targets, followed by `sam_amg_clip` and `dinov2_sam_siglip`.
- E2: large-vocabulary expansion is the clearest stress test. Scores drop sharply from Context59 to Context459 and from ADE150 to ADE847; the drop is not just a small mIoU effect, it is a vocabulary-scale failure mode.
- E3: rankings are stable across targets, but the worst target remains the large-vocabulary setting. This supports a story where method ordering is robust while absolute generalization is fragile.
- E4: fast dense baselines are orders of magnitude cheaper per image, while SAM/DINO+SAM variants buy quality with much higher latency. Runtime rows are usable for current analysis, but peak-memory/model-call instrumentation is still incomplete.

## E1 Standard Semantic Segmentation

Main table source: `runs/tables/e1_standard.csv`.

| Method | VOC20 | Context59 | ADE150 | COCO171 | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| maskclip | 37.79 | 8.15 | 1.88 | 4.35 | 13.04 |
| maskclip_attn | 37.73 | 8.35 | 1.82 | 4.08 | 13.00 |
| maskclip_attn_slide | 49.58 | 11.87 | 4.61 | 7.76 | 18.46 |
| sam_amg_clip | 65.99 | 17.84 | 12.59 | 12.36 | 27.20 |
| sam_amg_siglip | 65.60 | 19.66 | 15.34 | 14.81 | 28.85 |
| dinov2_sam_clip | 60.60 | 16.55 | 11.77 | 12.16 | 25.27 |
| dinov2_sam_siglip | 59.47 | 18.48 | 14.75 | 14.66 | 26.84 |

Interpretation: `sam_amg_siglip` has the best average, but the margin over `sam_amg_clip` and `dinov2_sam_siglip` is modest. The dense MaskCLIP family is much cheaper but far behind on ADE/COCO.

## E1 Official Baseline Context

Official/repo baseline table source: `runs/analysis/official_best_metrics.csv`.

| Method | VOC20 | Context59 | ADE150 | COCO171 | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| corrclip | 89.01 | 48.68 | 26.96 | 32.22 | 49.22 |
| proxyclip | 81.64 | 39.31 | 19.59 | 26.68 | 41.80 |
| naclip | 77.98 | 38.36 | 19.12 | 25.98 | 40.36 |
| resclip | 78.51 | 36.85 | 18.07 | 23.59 | 39.26 |
| sclip | 77.47 | 34.14 | 16.46 | 22.86 | 37.73 |

Interpretation: official dense/repo methods remain much stronger on the standard semantic targets. This is useful as context: the local proposal/naming adapters are not the main E1 winners, but they expose mechanism-level diagnostics and appendix hard-domain behavior.

## E2 Vocabulary Robustness

Main table source: `runs/tables/e2_vocab_robustness.csv`.

| Method | Ctx59 | Ctx459 | ADE150 | ADE847 | Avg vocab drop | MCMR@0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| maskclip | 8.15 | 1.22 | 1.88 | 0.49 | 4.16 | 0.805 |
| maskclip_attn | 8.35 | 1.25 | 1.82 | 0.50 | 4.21 | 0.825 |
| maskclip_attn_slide | 11.87 | 1.94 | 4.61 | 1.44 | 6.55 | 0.505 |
| sam_amg_clip | 17.84 | 4.00 | 12.59 | 5.04 | 10.69 | 0.600 |
| sam_amg_siglip | 19.66 | 4.97 | 15.34 | 7.24 | 11.40 | 0.552 |
| dinov2_sam_clip | 16.55 | 3.79 | 11.77 | 5.00 | 9.76 | 0.620 |
| dinov2_sam_siglip | 18.48 | 4.84 | 14.75 | 7.08 | 10.65 | 0.552 |

Interpretation: the largest story is not that one method slightly wins E2; it is that all methods are sensitive to vocabulary expansion. `sam_amg_siglip` and `dinov2_sam_siglip` retain the best large-vocabulary scores, but both still lose substantial performance from small to large vocabularies.

## E3 Cross-Target Generalization

Main local table source: `runs/tables/e3_generalization.csv`.

| Method | Target avg | Worst target mIoU |
| --- | ---: | ---: |
| maskclip | 13.04 | 1.88 |
| maskclip_attn | 13.00 | 1.82 |
| maskclip_attn_slide | 18.46 | 4.61 |
| sam_amg_clip | 27.20 | 12.36 |
| sam_amg_siglip | 28.85 | 14.81 |
| dinov2_sam_clip | 25.27 | 11.77 |
| dinov2_sam_siglip | 26.84 | 14.66 |

Official E3 diagnostic source: `runs/analysis/e3_generalization_official.csv`.

| Method | Target avg mIoU | Worst target | Worst target mIoU |
| --- | ---: | --- | ---: |
| corrclip | 36.24 | ade847 | 8.67 |
| proxyclip | 30.42 | ade847 | 6.90 |
| naclip | 29.21 | ade847 | 6.01 |
| resclip | 28.54 | ade847 | 6.49 |
| sclip | 27.21 | ade847 | 5.67 |

Interpretation: E3 gives a clean story: rankings are fairly stable, but ADE847 is the shared worst target. The weakness is structural, tied to large target vocabularies and long-tail label coverage.

## E4 Cost and Efficiency

Main local runtime source: `runs/tables/e4_cost.csv`.

| Method | Avg sec/image on E1 targets | Avg quality/sec |
| --- | ---: | ---: |
| maskclip | 0.081 | 4.523 |
| maskclip_attn | 0.020 | 6.916 |
| maskclip_attn_slide | 0.146 | 1.627 |
| sam_amg_clip | 4.249 | 0.063 |
| sam_amg_siglip | 5.747 | 0.048 |
| dinov2_sam_clip | 3.270 | 0.079 |
| dinov2_sam_siglip | 4.289 | 0.070 |

Official partial E4 source: `runs/analysis/e4_cost_partial_official.csv`.

Interpretation: dense methods are much faster. SAM/DINO+SAM methods are useful for region-centric analysis and hard-domain experiments, but their cost profile is much heavier. Current E4 runtime is enough for a cost-quality story; peak memory and model-call counts remain partial.

## Story Candidates From E1-E4

1. Aggregate mIoU hides structured failure. E2/E3 show that large vocabularies create systematic degradation, especially ADE847 and Context459.
2. Quality and cost move in opposite directions. `sam_amg_siglip` wins the local E1/E3 average but is far slower than dense MaskCLIP variants.
3. Official dense baselines are strong on standard semantic targets, while proposal/naming variants are more valuable for diagnostic and hard-domain analysis than for beating E1 leaderboards.
4. E4 isolated reruns are not required for the current summary. They would improve hardware-grade cost reporting, but the current cost table already supports the main efficiency tradeoff.

## Current Gaps

- E2 ambiguity table is not main-ready yet: `runs/tables/e2_ambiguity.csv` only has the debug row.
- Peak memory and model-call counts are incomplete in E4.
- Appendix/Table10 hard-domain results are still running and deliberately excluded here.
