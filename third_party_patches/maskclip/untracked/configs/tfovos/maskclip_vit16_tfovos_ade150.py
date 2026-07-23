# Official MaskCLIP config generated for TF-OVOS data paths.
# Uses the upstream MaskCLIP model and prompt embeddings; only data roots and
# evaluation splits are redirected to this workspace.

_base_ = ['../_base_/models/maskclip_vit16.py', '../_base_/datasets/ade20k.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py']

model = dict(
    decode_head=dict(
        num_classes=150,
        text_categories=150,
        text_channels=512,
        text_embeddings_path='pretrain/ade_ViT16_clip_text.pth',
        visual_projs_path='pretrain/ViT16_clip_weights.pth'))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    val=dict(data_root='/data/tianyi/TF-OVCOS/data/raw/ADEChallengeData2016', img_dir='images/validation', ann_dir='annotations/validation'),
    test=dict(data_root='/data/tianyi/TF-OVCOS/data/raw/ADEChallengeData2016', img_dir='images/validation', ann_dir='annotations/validation'))
