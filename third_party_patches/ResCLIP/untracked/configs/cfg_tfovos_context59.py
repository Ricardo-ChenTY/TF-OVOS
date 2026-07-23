_base_ = './base_config.py'

model = dict(name_path='./configs/cls_context59.txt',
    temp_thd=0.25,
    delete_same_entity=True,
    attn_rcs_weights=[2.0, 0.4],
    attn_sfr_weights=[1.8, 0.7],
)

dataset_type = 'PascalContext59Dataset'
data_root = '/data/tianyi/TF-OVCOS/data/official_mmseg/context59'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        ann_file='ImageSets/SegmentationContext/val.txt',
        reduce_zero_label=True,
        pipeline=test_pipeline))


# Official mmseg IoUMetric output. This preserves prediction label maps for
# proposal exploratory metrics without changing model inference or mIoU logic.
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/resclip/context59')

