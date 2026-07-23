# TF-OVOS E2 dataset config for FreeDA old-mmseg runtime.
_base_ = ["../custom_import.py"]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 448),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="FloatImage"),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

dataset_type = "ADE20KDataset847"
data_root = "/data/tianyi/TF-OVCOS/data/detectron2_refs/ADE20K_2021_17_01"

data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir="images_detectron2/validation",
        ann_dir="annotations_detectron2/validation",
        pipeline=test_pipeline,
    )
)

test_cfg = dict(mode="slide", stride=(224, 224), crop_size=(448, 448))
