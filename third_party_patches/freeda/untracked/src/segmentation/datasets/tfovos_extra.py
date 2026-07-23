import os
from pathlib import Path

from mmseg.datasets import DATASETS
from mmseg.datasets import CustomDataset


def _classes(name):
    cls_path = Path('/data/tianyi/TF-OVCOS/third_party/official_methods/Trident/configs') / name
    return tuple(line.strip() for line in cls_path.read_text(encoding='utf-8').splitlines() if line.strip())


@DATASETS.register_module(force=True)
class PascalContextDataset459(CustomDataset):
    CLASSES = _classes('cls_context459.txt')
    PALETTE = None

    def __init__(self, split=None, **kwargs):
        super(PascalContextDataset459, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.tif',
            split=split,
            reduce_zero_label=False,
            **kwargs,
        )
        assert os.path.exists(self.img_dir)


@DATASETS.register_module(force=True)
class ADE20KDataset847(CustomDataset):
    CLASSES = _classes('cls_ade20k847.txt')
    PALETTE = None

    def __init__(self, split=None, **kwargs):
        super(ADE20KDataset847, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.tif',
            split=split,
            reduce_zero_label=False,
            **kwargs,
        )
        assert os.path.exists(self.img_dir)
