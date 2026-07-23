# Official MaskCLIP config generated for TF-OVOS data paths.
# Uses the upstream MaskCLIP model and prompt embeddings; only data roots and
# evaluation splits are redirected to this workspace.

_base_ = ['../maskclip/maskclip_vit16_520x520_pascal_context_59.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root='/data/tianyi/TF-OVCOS/data/official_mmseg/context59', img_dir='JPEGImages', ann_dir='SegmentationClassContext', split='ImageSets/SegmentationContext/val.txt'),
    test=dict(data_root='/data/tianyi/TF-OVCOS/data/official_mmseg/context59', img_dir='JPEGImages', ann_dir='SegmentationClassContext', split='ImageSets/SegmentationContext/val.txt'))
