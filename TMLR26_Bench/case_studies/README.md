# Standalone Case Studies

These case studies are separate from the qualitative overview figure. Each one uses a two-row layout: full scene first, then a zoomed target region. The main paper should use one case study per promoted image-level observation.

| Case | Figure | Observation supported | Main message |
| --- | --- | --- | --- |
| Combined ADE-847 large-vocab | `../figures/case_study_large_vocab_combined.pdf` | High mIoU does not imply large-vocabulary coverage. | Independent `shelf` and `jacket` examples show that expanded vocabularies create label competition even when the region is visually meaningful. |
| ADE-150 / box | `../figures/case_study_mismatch_box_bookcase.pdf` | Localization and naming errors should be separated. | Methods alternate between `box`, `bookcase`, and `cabinet`, showing that a plausible mask can still carry the wrong semantic name. |
| ADE-150 / bus | `../figures/case_study_small_object_bus_independent.pdf` | Small components are a systematic weak point. | The small `bus` target is absorbed into surrounding context and often renamed as `road`. |

Optional layout fallback files:

| Case | Figure | Observation supported | Main message |
| --- | --- | --- | --- |
| ADE-847 / shelf | `../figures/case_study_large_vocab_shelf_ade847.pdf` | High mIoU does not imply large-vocabulary coverage. | Standalone source panel for the combined E2 case. |
| ADE-847 / jacket | `../figures/case_study_large_vocab_jacket_ade847.pdf` | High mIoU does not imply large-vocabulary coverage. | Standalone source panel for the combined E2 case. |

The LaTeX section that uses these figures is `../sections/case_studies.tex`, and it is included from `../sections/04_experiments_and_analysis.tex`. The current generator is `../scripts/make_independent_case_studies.py`.
