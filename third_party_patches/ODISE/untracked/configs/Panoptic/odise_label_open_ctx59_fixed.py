# Local eval-only config: run only fixed PASCAL Context-59 for TF-OVCOS.
from .odise_label_coco_50e import model, dataloader, train, optimizer, lr_multiplier
from .odise_label_coco_50e import _ctx59_eval

if "evaluator" in dataloader:
    del dataloader.evaluator

# The dataset registry points ctx59_sem_seg_val to annotations_ctx59_fixed.
dataloader.extra_task = dict(eval_ctx59=_ctx59_eval)
