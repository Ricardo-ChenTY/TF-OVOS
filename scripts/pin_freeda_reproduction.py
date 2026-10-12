"""Pin all retrieved prototype inputs and prove a clean reload reproduces new PNGs."""
import json,os,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from e4_special_backends import FreeDA
from e4_full_common import sha,save,load_rows,utc
R=Path('/data/tianyi/TF-OVOS');B=R/'runs/rebuttal/closure_20261011';O=B/'freeda_reproduction';O.mkdir(exist_ok=True)
actual_load=np.load;accessed={}
def tracked_load(file,*a,**kw):
 result=actual_load(file,*a,**kw)
 if isinstance(file,(str,os.PathLike)) and 'prototype_embeddings' in str(file):accessed[str(Path(file).resolve())]=sha(file)
 return result
np.load=tracked_load
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
try:model=FreeDA('freeda','coco_stuff171',O)
finally:np.load=actual_load
for name in ['text_embedding','proto_embedding']:np.save(O/(name+'.npy'),getattr(model.model,name).detach().cpu().numpy())
save(O/'retrieved_prototypes_sha256.json',accessed)
rows=load_rows(R/'data/manifests/coco_stuff171_val.jsonl');checks=[]
for row in rows[:5]:
 for key in ['image_path','mask_path']:row[key]=str((R/'data/manifests'/row[key]).resolve()) if not Path(row[key]).is_absolute() else row[key]
 with Image.open(row['image_path']) as im:rgb=im.convert('RGB')
 with torch.inference_mode():pred=model.infer(row,rgb)
 ref=B/'freeda_full/predictions'/(row['image_id']+'.png');other=np.asarray(Image.open(ref));assert np.array_equal(pred,other)
 checks.append(dict(image_id=row['image_id'],exact_match=True,reference_sha256=sha(ref)))
repo=Path(model.meta['repo']);sources=[repo/'models/freeda/freeda.py',repo/'segmentation/evaluation/freeda_builder.py',repo/'segmentation/evaluation/freeda_seg.py',repo/'segmentation/configs/_base_/datasets/stuff.py',R/'scripts/e4_special_backends.py',Path(__file__).resolve(),Path('/data/tianyi/TF-OVCOS/runs/freeda/coco_stuff171/config.json'),Path('/data/tianyi/TF-OVCOS/runs/freeda_artifact/coco_stuff171/config.json'),Path('/data/tianyi/TF-OVCOS/runs/freeda/coco_stuff171/log.txt'),Path('/data/tianyi/TF-OVCOS/runs/freeda_artifact/coco_stuff171/log.txt')]
weights=[]
for rec in json.loads((R/'runs/rebuttal/e4_20261009/new_checkpoint_and_index_sha256.json').read_text()):
 if any(s in rec['path'] for s in ['ViT-L-14.pt','vit_large_patch14_dinov2','knn.index']):
  assert sha(rec['path'])==rec['sha256'];weights.append(rec)
assert len(weights)==3
save(B/'freeda_provenance.json',dict(status='verified',updated_utc=utc(),native_class_names=model.meta['class_names'],settings=model.meta,weights_and_index=weights,source_sha256={str(p):sha(p) for p in sources},retrieved_prototype_count=len(accessed),retrieved_prototypes_manifest=str(O/'retrieved_prototypes_sha256.json'),retrieved_prototypes_manifest_sha256=sha(O/'retrieved_prototypes_sha256.json'),embedding_sha256={name:sha(O/(name+'.npy')) for name in ['text_embedding','proto_embedding']},clean_reload_checks=checks,scope='Pins all actually retrieved prototype files, backbone, CLIP checkpoint, index, inference configuration and code; five independent fresh-load predictions match full repaired outputs.'))
print('PROVENANCE VERIFIED',len(accessed),flush=True)
