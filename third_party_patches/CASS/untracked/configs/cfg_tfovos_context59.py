_base_ = "./cfg_context59.py"

data_root = "/data/tianyi/TF-OVCOS/data/raw/VOCdevkit/VOC2010"
test_dataloader = dict(dataset=dict(data_root=data_root))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/context59')

