# CorrCLIP real Observation-5 diagnostic pilot

This is a real-call pilot, not a replacement for the existing dense-label-map proxy table. It evaluates CorrCLIP only on the first 250 manifest records per requested dataset.

## Results

| Metric | Context-459 | ADE-847 | Unweighted mean | Old six-dataset proxy | Mean minus proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| GT-region naming Top-1 | 0.227 | 0.160 | 0.193 | 0.487 | -0.294 |
| GT-text localization IoU | 0.355 | 0.218 | 0.287 | 0.403 | -0.116 |
| Proposal oracle IoU | 0.668 | 0.624 | 0.646 | 0.605 | +0.041 |
| Proposal Recall@0.5 | 0.703 | 0.679 | 0.691 | 0.635 | +0.056 |

The old values (0.487 / 0.403 / 0.605 / 0.635) are the disclosed proxy values: an unweighted six-dataset average reconstructed from CorrCLIP dense argmax maps. They are not directly like-for-like with this two-dataset, real-call pilot.

## What was actually run

- **Backbone:** MetaCLIP FullCC `ViT-B-16-quickgelu`, using CorrCLIP's cached `b16_fullcc2.5b.pt` weights. Region naming uses a conventional global open_clip ViT-B/16-quickgelu encoder loaded strictly from that same checkpoint; CorrCLIP's vendored vision tower cannot serve a global crop embedding because it is modified to require DINO correlation inputs. Localization uses CorrCLIP's vendored DINO ViT-B/8 + correlation-refined dense encoder directly.
- **Prompts:** CorrCLIP's own `openai_imagenet_template` ensemble, mean-normalized per dataset vocabulary class. The repository's CorrCLIP vocabulary files match the benchmark vocabularies byte-for-byte apart from their final newline.
- **Naming:** for each present semantic class, GT pixels outside that class are blacked out, the tight bounding-box crop is CLIP-encoded, and its cosine top-1 is evaluated over the full fixed dataset vocabulary.
- **Localization:** the correlation-refined dense image features are computed once per image, then scored against the *full* candidate vocabulary (not just the one correct class). The predicted region for a given GT class is the set of pixels where that class wins the argmax against every other candidate class, using CorrCLIP's own real dense scores (freshly computed here, not read from a saved prediction map). An earlier version of this pilot scored only the single correct class's prompt and Otsu-thresholded that one map; it was rejected after a manual check showed predicted regions collapsing to under 0.3% of the image regardless of GT size (3.5-22%), because a single-prompt similarity map has too narrow/unimodal a value range for Otsu to split meaningfully. The argmax version does not fully isolate localization from naming competition -- a class that never wins the full argmax anywhere gets IoU=0 here even if CorrCLIP's patch-level correlation for it is locally reasonable -- but it is a real, non-degenerate measurement rather than a broken one.
- **Proposals:** SAM ViT-H automatic-mask generation from `weights/sam_vit_h_4b8939.pth`, with the exact parameters used by `SamAmgClipAdapter`: points-per-side 32, predicted-IoU threshold 0.86, stability threshold 0.92, and min region area 400. For every semantic GT region, oracle IoU is the largest IoU of one generated SAM mask; Recall@0.5 is the fraction with best IoU >= 0.5.

## Coverage and failures

| Dataset | Requested | Attempted | Fully processed | Naming regions | Localization regions | Proposal regions | SAM masks | Empty-SAM images | CorrCLIP saved-mask fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Context-459 | 250 | 250 | 250 | 1492 | 1492 | 1492 | 21173 | 0 | 0 |
| ADE-847 | 250 | 250 | 250 | 3026 | 3026 | 3026 | 24038 | 0 | 0 |

## Caveats

This pilot is limited to roughly 250 images per dataset (first manifest order), two large-vocabulary datasets, and CorrCLIP alone. Semantic regions are class unions, not separately annotated object instances. The proposal results measure generic SAM ViT-H coverage, not CorrCLIP's saved SAM2 masks or a learned proposal method. The localization result uses full-vocabulary argmax (see above) rather than a pure single-class score, so it is not a fully naming-independent localization measurement -- classes that never win the argmax anywhere get IoU=0 by construction. These measurements establish that the four quantities were obtained with actual CLIP/CorrCLIP/SAM computations on this subset; they do not establish full-dataset, method-general, or paper-final claims.
