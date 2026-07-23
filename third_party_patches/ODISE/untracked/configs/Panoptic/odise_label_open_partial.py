# Local eval-only config: official ODISE model with only datasets available in DETECTRON2 layout.
from .odise_label_coco_50e import model, dataloader, train, optimizer, lr_multiplier
from .odise_label_coco_50e import _ade847_eval, _ctx59_eval, _ctx459_eval, _pas21_eval

# Avoid primary COCO/ADE panoptic evals because panoptic json/source files are not present locally.
# Keep dataloader.test: the official wrapper metadata interpolates test.dataset.names.
if "evaluator" in dataloader:
    del dataloader.evaluator

dataloader.extra_task = dict(
    eval_pas21=_pas21_eval,
    eval_ctx59=_ctx59_eval,
    eval_ctx459=_ctx459_eval,
    eval_ade847=_ade847_eval,
)
