data_root = '/data/tianyi/TF-OVCOS/data/official_mmseg/context59'
dataset_type = 'PascalContext59Dataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=2000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(interval=1, type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    clip_path='ViT-B/16',
    multi_end_idx=10,
    multi_start_idx=3,
    name_path='./configs/cls_context59.txt',
    post_adjust_idx=3,
    pre_adjust_idx=8,
    res_cls=0.3,
    type='SCCLIPForSegmentation')
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='ImageSets/SegmentationContext/val_subset300.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClassContext'),
        data_root='/data/tianyi/TF-OVCOS/data/official_mmseg/context59',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2048,
                336,
            ), type='Resize'),
            dict(reduce_zero_label=True, type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        type='PascalContext59Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ],
    output_dir=
    '/data/tianyi/TF-OVCOS/runs/tuning_ablation/scclip_context59/preds_0.3',
    type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        2048,
        336,
    ), type='Resize'),
    dict(reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/data/tianyi/TF-OVCOS/runs/tuning_ablation/scclip_context59/work_0.3'
