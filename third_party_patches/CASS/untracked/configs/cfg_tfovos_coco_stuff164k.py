_base_ = "./cfg_coco_stuff164k.py"

data_root = "/data/tianyi/TF-OVCOS/data/official_mmseg/coco_stuff171"
test_dataloader = dict(dataset=dict(data_root=data_root))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/cass/coco_stuff164k')

