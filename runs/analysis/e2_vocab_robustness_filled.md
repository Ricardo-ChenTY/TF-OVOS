# E2 Vocabulary Robustness Filled

| method | context59_miou | context459_miou | context459_ziou | ade150_miou | ade847_miou | ade847_ziou | delta_vocab | mcmr_05_e2 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MaskCLIP | 8.15 | 1.22 | 64.1 | 1.88 | 0.49 | 64.7 | 4.16 | 0.843 | filled from existing artifacts |
| MaskCLIP-Attn | 8.35 | 1.25 | 63.4 | 1.82 | 0.50 | 64.7 | 4.21 | 0.858 | filled from existing artifacts |
| MaskCLIP-Attn-Slide | 11.87 | 1.94 | 50.5 | 4.61 | 1.44 | 38.5 | 6.55 | 0.541 | filled from existing artifacts |
| SCLIP | 34.14 | 6.68 | 39.7 | 16.46 | 5.67 | 39.2 | 19.12 | 0.332 | filled from existing artifacts |
| NACLIP | 38.36 | 7.79 | 38.6 | 19.12 | 6.01 | 36.3 | 21.84 | 0.345 | filled from existing artifacts |
| CLIPtrase | 34.58 | 9.95 | 54.2 | 17.04 | 5.89 | 61.8 | 17.89 | 0.455 | ZIoU map-proxy; MCMR from dense-map diagnostics |
| SC-CLIP | 40.21 | 8.34 | 41.5 | 20.06 | 7.46 | 40.8 | 22.23 | 0.362 | filled from existing artifacts |
| ResCLIP | 36.85 | 7.70 | 39.4 | 18.07 | 6.49 | 39.1 | 20.37 | 0.336 | filled from existing artifacts |
| ProxyCLIP | 39.31 | 8.41 | 42.4 | 19.59 | 6.90 | 42.0 | 21.80 | 0.397 | filled from existing artifacts |
| CorrCLIP | 48.68 | 11.88 | 44.2 | 26.96 | 8.67 | 49.9 | 27.54 | 0.421 | filled from existing artifacts |
| Trident | 40.99 | 8.97 | 44.2 | 20.91 | 7.45 | 47.1 | 22.74 | 0.432 | filled from existing artifacts |
| CASS | 40.27 | 8.38 | 43.2 | 20.32 | 7.19 | 44.7 | 22.51 | 0.379 | filled from existing artifacts |
| FreeDA | 43.42 | 8.97 | 41.2 | 23.19 | 6.39 | 40.9 | 25.62 | 0.476 | Ctx459/ADE847 mIoU+ZIoU from our own reconstruction, not FreeDA's own log (log is internally truncated past a point in the vocabulary; see plan.md 2.2) |
| SAM-AMG + CLIP | 17.84 | 4.00 | 42.5 | 12.59 | 5.04 | 45.3 | 10.69 | 0.776 | filled from existing artifacts |
| SAM-AMG + SigLIP | 19.66 | 4.97 | 41.8 | 15.34 | 7.24 | 38.9 | 11.40 | 0.713 | filled from existing artifacts |
| DINOv2 + SAM + CLIP | 16.55 | 3.79 | 42.5 | 11.77 | 5.00 | 44.7 | 9.76 | 0.776 | filled from existing artifacts |
| DINOv2 + SAM + SigLIP | 18.48 | 4.84 | 41.2 | 14.75 | 7.08 | 38.7 | 10.65 | 0.711 | filled from existing artifacts |
| SAN (ref.) | 52.43 | 12.75 | 31.6 | 27.56 | 10.24 | 51.3 | 28.50 | 0.318 | ZIoU from log; MCMR from converted SAN maps |
