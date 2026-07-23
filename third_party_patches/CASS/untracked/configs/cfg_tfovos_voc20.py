_base_ = "./cfg_voc20.py"

data_root = "/data/tianyi/TF-OVCOS/data/raw/VOCdevkit/VOC2012"
test_dataloader = dict(dataset=dict(data_root=data_root))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/voc20')

