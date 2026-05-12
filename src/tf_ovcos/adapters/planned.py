from __future__ import annotations

from pathlib import Path

from tf_ovcos.adapters.base import MethodAdapter
from tf_ovcos.data import Prediction, Sample


class PlannedAdapter(MethodAdapter):
    runnable = False
    setup_hint = "This adapter is planned but not implemented yet."

    def predict_one(self, sample: Sample, vocabulary: list[str], output_dir: Path) -> Prediction:
        raise NotImplementedError(self.setup_hint)


class GroundingDinoSamAdapter(PlannedAdapter):
    name = "groundingdino_sam"
    setup_hint = (
        "groundingdino_sam needs the Grounded-Segment-Anything repo, "
        "GroundingDINO weights, SAM weights, CUDA PyTorch, and model-loading code."
    )


class SamAmgClipAdapter(PlannedAdapter):
    name = "sam_amg_clip"
    setup_hint = (
        "sam_amg_clip needs Segment Anything, CLIP/SigLIP scoring, model weights, "
        "and proposal-ranking code."
    )


class SclipAdapter(PlannedAdapter):
    name = "sclip"
    setup_hint = "sclip needs the external SCLIP repo, CUDA PyTorch, weights, and dense-map conversion code."


class ProxyClipAdapter(PlannedAdapter):
    name = "proxyclip"
    setup_hint = "proxyclip needs the external ProxyCLIP repo, CUDA PyTorch, weights, and dense-map conversion code."
