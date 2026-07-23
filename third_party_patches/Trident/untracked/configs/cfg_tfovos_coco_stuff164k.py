_base_ = "./cfg_coco_stuff164k.py"

model = dict(
    sam_ckpt="/data/tianyi/TF-OVCOS/weights/sam_vit_b_01ec64.pth",
    sam_model_type="vit_b",
)

data_root = "/data/tianyi/TF-OVCOS/data/official_mmseg/coco_stuff171"
test_dataloader = dict(dataset=dict(data_root=data_root))

test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'], output_dir='/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/trident/coco_stuff164k')

