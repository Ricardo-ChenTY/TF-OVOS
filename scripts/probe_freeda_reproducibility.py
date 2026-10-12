"""Read-only native FreeDA vs archived PNG correspondence on fixed image IDs."""
import json,os,sys,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/data/tianyi/TF-OVOS');OUT=ROOT/'runs/rebuttal/closure_20261011/freeda_probe';OUT.mkdir(parents=True,exist_ok=True)
from e4_special_backends import FreeDA
import torch
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
rows=[json.loads(x) for x in (ROOT/'data/manifests/coco_stuff171_val.jsonl').read_text().splitlines()]
selected=[rows[i] for i in [0,250,1000,2500,4999]]
for row in selected:
 for key in ['image_path','mask_path']:
  row[key]=str((ROOT/'data/manifests'/row[key]).resolve()) if not Path(row[key]).is_absolute() else row[key]
model=FreeDA('freeda','coco_stuff171',OUT)
(OUT/'settings.json').write_text(json.dumps(model.meta,indent=2,default=str))
counts=np.zeros((171,256),dtype=np.int64);results=[]
for row in selected:
 with Image.open(row['image_path']) as im:rgb=im.convert('RGB')
 with torch.no_grad():pred=model.infer(row,rgb).astype(np.int32)
 oldpath=Path('/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/coco_stuff164k')/(row['image_id']+'.png')
 old=np.array(Image.open(oldpath)).astype(np.int32)
 assert pred.shape==old.shape
 counts+=np.bincount((pred*256+old).ravel(),minlength=171*256).reshape(171,256)
 Image.fromarray(pred.astype(np.uint8)).save(OUT/(row['image_id']+'.png'))
 r=dict(image_id=row['image_id'],equal_pixels=int((pred==old).sum()),pixels=pred.size,old_rejection_pixels=int((old==255).sum()),native_min=int(pred.min()),native_max=int(pred.max()))
 results.append(r);print(r,flush=True)
 np.save(OUT/'native_vs_archive_counts.npy',counts)
 (OUT/'comparison.json').write_text(json.dumps(results,indent=2))
print('DOMINANT MAPPING',[(i,int(x.argmax()),int(x.sum()),float(x.max()/x.sum())) for i,x in enumerate(counts) if x.sum()],flush=True)
