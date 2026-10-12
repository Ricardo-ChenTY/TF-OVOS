"""Independent full COCO confusion for archived FreeDA, including all raw labels."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from PIL import Image
R=Path('/data/tianyi/TF-OVOS');out=R/'runs/rebuttal/closure_20261011/freeda_saved';out.mkdir(exist_ok=True)
rows=[json.loads(x) for x in (R/'data/manifests/coco_stuff171_val.jsonl').read_text().splitlines()]
conf=np.zeros((171,256),dtype=np.int64);records=[]
for i,r in enumerate(rows):
 p=Path('/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/coco_stuff164k')/(r['image_id']+'.png');gp=(R/'data/manifests'/r['mask_path']).resolve()
 g=np.asarray(Image.open(gp)).astype(np.int64);pred=np.asarray(Image.open(p)).astype(np.int64);assert g.shape==pred.shape
 valid=g<171;conf+=np.bincount(g[valid]*256+pred[valid],minlength=171*256).reshape(171,256)
 records.append(dict(image_id=r['image_id'],prediction_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),gt_sha256=hashlib.sha256(gp.read_bytes()).hexdigest()))
 if (i+1)%1000==0:print(i+1,flush=True)
np.save(out/'raw_confusion.npy',conf)
scores={}
for shift in [-1,0,1]:
 c=np.zeros((171,172),dtype=np.int64)
 for raw in range(256):
  label=raw+shift;c[:,label if 0<=label<171 else 171]+=conf[:,raw]
 tp=np.diag(c[:,:171]);u=c.sum(1)+c[:,:171].sum(0)-tp;scores[str(shift)]=float((tp/u).mean()*100)
(out/'audit.json').write_text(json.dumps(dict(images=len(rows),offset_miou_percent=scores,raw_pred_counts=conf.sum(0).tolist(),records=records),indent=2))
print(scores,flush=True)
