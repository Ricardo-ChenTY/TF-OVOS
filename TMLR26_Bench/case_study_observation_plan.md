# Case Study Plan for TF-OVOS Observations

更新时间：2026-05-28

这份文件用于决定论文里放哪些 case study，以及它们分别支撑哪个 observation。建议不要把 case study 当作新指标，而是作为 Table 6、diagnostic probes、promoted observations 的可视化证据。

## Recommended Main-Paper Case Studies

原则：一个 case study 对应一个 promoted observation。当前三个主 case study 都是 independent examples，不复用 qualitative overview 里的 panel。

### Case Study 1: combined large-vocabulary coverage failure

- Figure file: `figures/case_study_large_vocab_combined.pdf`
- Source layout: two stacked standalone case-study panels, each with full scene and zoom target rows
- Dataset / image A: ADE-847, `ADE_val_00000456.jpg`
- Target A: `shelf`
- Dataset / image B: ADE-847, `ADE_val_00000255.jpg`
- Target B: `jacket`
- Observation supported: high mIoU does not imply large-vocabulary coverage.
- What it shows: methods rename visible `shelf` / `jacket` regions as contextual or competing labels such as `floor`, `flower`, or `skylight`.
- Paper role: use two image-level examples as one case study for the E2 large-vocabulary observation.

Suggested paragraph:

> The combined ADE-847 case shows two forms of large-vocabulary coverage failure. A visible shelf is renamed as contextual scene objects, and a jacket region is consistently mapped to a competing label. Together, these examples are the visual counterpart of the ZIoU diagnostic, where many large-vocabulary classes receive no effective coverage.

### Case Study 2: mask-category mismatch

- Figure file: `figures/case_study_mismatch_box_bookcase.pdf`
- Dataset / image: ADE-150, `ADE_val_00000029.jpg`
- Target: `box`
- Observation supported: localization and naming failures separate cleanly.
- What it shows: methods predict visually plausible indoor structures but alternate between labels such as `bookcase`, `cabinet`, and `box`.
- Paper role: explain why MCMR is necessary and why mIoU alone cannot distinguish wrong-mask from wrong-name errors.

Suggested paragraph:

> The box/bookcase case illustrates a mask-category mismatch: methods attend to plausible indoor object regions, but semantic assignment is unstable across nearby vocabulary labels. This supports reporting MCMR alongside mIoU because a region can be spatially meaningful while still semantically incorrect.

### Case Study 3: small-component degradation

- Figure file: `figures/case_study_small_object_bus_independent.pdf`
- Dataset / image: ADE-150, `ADE_val_00000735.jpg`
- Target: `bus`
- Observation supported: small components remain a universal weak point.
- What it shows: the small target bus is absorbed into surrounding road/context labels.
- Paper role: visually support the component-scale result where small/large IoU ratio is about 0.052 on average.

Suggested paragraph:

> The bus case shows that small objects can disappear into surrounding context. Even methods with strong compact-vocabulary mIoU assign the small bus region to road-like context. This visualizes the component-scale diagnostic: small-region IoU collapses much more sharply than large-region IoU.

## Optional Appendix / Backup Files

The combined E2 figure is built from two standalone panels:

- `figures/case_study_large_vocab_shelf_ade847.pdf`
- `figures/case_study_large_vocab_jacket_ade847.pdf`

Use the standalone versions only if the combined figure is too tall for the main layout.

## Quantitative Observation Figures

These are not image-level case studies, but they can be paired with the case studies to make the observations stronger.

### Observation 1: Large-vocabulary class silence

- Figure file copied to paper: `figures/fig_vocab_collapse_smooth.pdf`
- Supports: high mIoU does not imply broad vocabulary coverage.
- Use with Table 6: CorrCLIP has the strongest strict TF large-vocab mIoU but still has 44.2% Context-459 and 49.9% ADE-847 zero-IoU classes.

### Observation 2: mIoU-coverage mismatch

- Figure file copied to paper: `figures/fig_miou_vs_class_collapse_smooth.pdf`
- Supports: stronger mIoU can coexist with high zero-IoU collapse.
- Use carefully: phrase as concentration of gains rather than saying mIoU is bad.

### Observation 3: Small-component collapse

- Figure file copied to paper: `figures/fig_component_scale_smooth.pdf`
- Supports: small object failures are systematic, not just one visual example.
- Pair with: bus case study.

### Observation 4: Efficiency Pareto

- Figure file copied to paper: `figures/fig_efficiency_pareto.pdf`
- Supports: practical ranking changes under runtime constraints.
- Main points: CorrCLIP is quality leader; Trident is a speed-quality point; proposal+naming methods are slower.

### Observation 5: Diagnostic decomposition

- Figure file copied to paper: `figures/fig_diagnostic_decomposition.pdf`
- Supports: naming, localization, and proposal ceilings are separable.
- Pair with: box/bookcase case study.

## Recommended Placement

Main paper:

- Use the three main standalone case-study figures, one per promoted image-level observation.
- Keep the existing qualitative comparison figure as a compact overview, but cite the standalone case studies when making the observation-level claims.
- Use `case_study_large_vocab_combined` for the E2 observation; use the two standalone E2 files only as layout fallbacks.

Appendix:

- If the main paper is crowded, keep only two case studies in the main text and move the remaining standalone figures to the appendix.
- Keep the quantitative PDF figures as appendix figures unless the main paper has room for one combined observation figure.

## LaTeX Snippet

A ready-to-use section is available at:

`sections/case_studies.tex`

It is already included from `sections/04_experiments_and_analysis.tex`.
