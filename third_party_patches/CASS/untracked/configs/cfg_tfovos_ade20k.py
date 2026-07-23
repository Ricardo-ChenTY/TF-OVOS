_base_ = "./cfg_ade20k.py"

data_root = "/data/tianyi/TF-OVCOS/data/raw/ADEChallengeData2016"
test_dataloader = dict(dataset=dict(data_root=data_root))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/ade20k')

