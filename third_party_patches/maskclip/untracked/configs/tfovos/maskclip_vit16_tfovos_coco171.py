# Official MaskCLIP config generated for TF-OVOS data paths.
# Uses the upstream MaskCLIP model and prompt embeddings; only data roots and
# evaluation splits are redirected to this workspace.

_base_ = ['../maskclip/maskclip_vit16_512x512_coco-stuff164k.py']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root='/data/tianyi/TF-OVCOS/data/official_mmseg/coco_stuff171', img_dir='images/val2017', ann_dir='annotations/val2017'),
    test=dict(data_root='/data/tianyi/TF-OVCOS/data/official_mmseg/coco_stuff171', img_dir='images/val2017', ann_dir='annotations/val2017'))
