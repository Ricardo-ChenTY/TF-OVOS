# Case Study Notes for TF-OVOS

更新时间：2026-05-28

这个文档记录当前论文正文使用的 independent case studies。它们不复用 qualitative overview 里的 person/scarf/signboard/bus panel，而是从 ADE20K validation rows 重新选样本，并用本地 official prediction maps 生成 PDF。

## Main Case Study Mapping

| Case study | Figure | Observation | Role in paper |
| --- | --- | --- | --- |
| Combined ADE-847 large-vocabulary coverage failure | `figures/case_study_large_vocab_combined.pdf` | High mIoU does not imply large-vocabulary coverage. | 用 `shelf` 和 `jacket` 两个 ADE-847 例子说明 compact-vocabulary 表现不能代表 large-vocabulary coverage。 |
| ADE-150 box/bookcase mismatch | `figures/case_study_mismatch_box_bookcase.pdf` | Localization and naming failures separate cleanly. | 说明 mask 可能覆盖合理区域，但 vocabulary naming 在相近物体间不稳定。 |
| ADE-150 small bus | `figures/case_study_small_object_bus_independent.pdf` | Small components remain a universal weak point. | 说明小目标容易被上下文吸收，支持 component-scale diagnostic。 |

正文 LaTeX 文件：

- Section: `sections/case_studies.tex`
- Included from: `sections/04_experiments_and_analysis.tex`
- Generated figures live in: `figures/`
- Figure-generation helper: `scripts/make_independent_case_studies.py`

## Case Study 1: Combined ADE-847 Large-Vocabulary Coverage Failure

### Figure files

- Main combined figure: `figures/case_study_large_vocab_combined.pdf`
- Standalone source panel A: `figures/case_study_large_vocab_shelf_ade847.pdf`
- Standalone source panel B: `figures/case_study_large_vocab_jacket_ade847.pdf`

### LaTeX label

```latex
\label{fig:case-study-large-vocab-combined}
```

### Observation supported

High mIoU does not imply large-vocabulary coverage.

### Quantitative anchor

Use this case together with E2 / Table 6:

- CorrCLIP remains the strongest strict TF row under large vocabularies.
- But CorrCLIP still has low large-vocabulary mIoU: 11.88 on Context-459 and 8.67 on ADE-847.
- CorrCLIP also has high zero-IoU class-collapse rates: 44.2% on Context-459 and 49.9% on ADE-847.
- This means high compact-vocabulary performance can coexist with poor broad-vocabulary coverage.

### Subcase A: ADE-847 / shelf

- Dataset: ADE-847
- Image id: `ADE_val_00000456.jpg`
- Source raw object name: `shelves`
- Target class: `shelf`
- Visual setting: indoor room with shelf-like furniture and surrounding floor/wall objects.
- Main failure: the target is renamed as broad contextual labels.
- Representative predictions:
  - GT: `shelf[gt]`
  - CorrCLIP: `floor[X]`
  - NACLIP: `fireplace[X]`
  - ProxyCLIP: `flower[X]`
  - SCLIP: `field[X]`

Interpretation:

The target is object-like and visible, but broad-vocabulary prediction prefers surrounding scene context or nearby objects. This visualizes E2 class silence: the model can segment a meaningful area without assigning the intended ADE-847 name.

### Subcase B: ADE-847 / jacket

- Dataset: ADE-847
- Image id: `ADE_val_00000255.jpg`
- Source raw object name: `jacket`
- Target class: `jacket`
- Visual setting: indoor doorway / clothing region.
- Main failure: a fine-grained target is absorbed into a competing scene/surface label.
- Representative predictions:
  - GT: `jacket[gt]`
  - CorrCLIP: `skylight[X]`
  - NACLIP: `skylight[X]`
  - ProxyCLIP: `skylight[X]`
  - SCLIP: `skylight[X]`

Interpretation:

The error is not only a missed boundary. Multiple methods converge on the same wrong ADE-847 label, showing that expanded vocabularies introduce strong semantic competition for fine-grained classes.

### Paper-ready paragraph

```latex
\paragraph{Case 1: high mIoU does not imply large-vocabulary coverage.}
Figure~\ref{fig:case-study-large-vocab-combined} combines two ADE-847 examples that are not part of the qualitative overview. In the first, the target class is \emph{shelf}, but methods rename the target region using scene-context labels such as \emph{floor} or \emph{flower}. In the second, the target class is \emph{jacket}, yet several methods assign the region to \emph{skylight}. Together, these examples are the image-level counterpart of the large-vocabulary coverage observation in Table~\ref{tab:promoted-observations}: a method may segment a meaningful region while still failing to recover the intended category under a broad candidate vocabulary.
```

## Case Study 2: ADE-150 Box / Bookcase Mask-Category Mismatch

### Figure files

- Main figure: `figures/case_study_mismatch_box_bookcase.pdf`

### LaTeX label

```latex
\label{fig:case-study-mismatch-box-bookcase}
```

### Observation supported

Localization and naming failures separate cleanly.

### Quantitative anchor

Use this case together with MCMR and diagnostic probes:

- MCMR@0.5 measures cases where predicted masks can overlap meaningful regions but category assignment is wrong.
- The diagnostic table separates naming, localization, and proposal ceilings.
- This case explains why mIoU alone is not enough: it merges wrong-mask and wrong-name errors.

### Case metadata

- Dataset: ADE-150
- Image id: `ADE_val_00000029.jpg`
- Source raw object name: `boxes`
- Target class: `box`
- Visual setting: cluttered indoor scene with shelf/bookcase-like background structure.
- Main failure: methods select plausible nearby structures but disagree on the semantic name.
- Representative predictions:
  - GT: `box[gt]`
  - CorrCLIP: `bookcase[X]`
  - NACLIP: `cabinet[X]`
  - ProxyCLIP: `box[X]`
  - SCLIP: `cabinet[X]`

Interpretation:

The methods are not simply producing empty masks. They attend to visually plausible indoor objects, but the semantic category is unstable. This supports separating localization and naming failures in the evaluation.

## Case Study 3: ADE-150 Small Bus Component Failure

### Figure files

- Main figure: `figures/case_study_small_object_bus_independent.pdf`

### LaTeX label

```latex
\label{fig:case-study-small-object-bus}
```

### Observation supported

Small components remain a universal weak point.

### Quantitative anchor

Use this case together with component-scale diagnostics:

- CorrCLIP component IoU drops from 0.370 on large components to 0.113 on medium components and 0.018 on small components.
- SCLIP similarly drops from 0.257 to 0.063 to 0.012.
- The visual case explains what the diagnostic means: small targets can be overwhelmed by surrounding context.

### Case metadata

- Dataset: ADE-150
- Image id: `ADE_val_00000735.jpg`
- Source raw object name: `bus`
- Target class: `bus`
- Visual setting: outdoor waterfront / skyline scene with a small target component.
- Main failure: the target is absorbed into road/context labels.
- Representative predictions:
  - GT: `bus[gt]`
  - CorrCLIP: `road[X]`
  - NACLIP: `road[X]`
  - ProxyCLIP: `road[X]`
  - SCLIP: `road[X]`

Interpretation:

The failure is spatial as much as semantic. The target bus is small compared with the surrounding scene, so patch-level evidence and dense-map evidence are dominated by contextual regions. This shows why component-level readouts are necessary alongside image-level or class-level mIoU.

## Recommended Text Flow

Suggested order in the paper:

1. Introduce Table~\ref{tab:promoted-observations}.
2. State that the table promotes three image-level observations.
3. Show the combined ADE-847 case for large-vocabulary coverage failure.
4. Show the box/bookcase case for naming/localization separation.
5. Show the bus case for small-component degradation.
6. Leave efficiency and proposal-ceiling observations to aggregate diagnostic figures and tables rather than image-level cases.

## Files Checklist

Required for the current main paper:

- `figures/case_study_large_vocab_combined.pdf`
- `figures/case_study_mismatch_box_bookcase.pdf`
- `figures/case_study_small_object_bus_independent.pdf`
- `sections/case_studies.tex`

Useful fallback files:

- `figures/case_study_large_vocab_shelf_ade847.pdf`
- `figures/case_study_large_vocab_jacket_ade847.pdf`
