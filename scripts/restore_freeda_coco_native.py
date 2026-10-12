"""Targeted full native FreeDA COCO rerun after lossy exported-label audit.

Resumable PNG+hash journal; predictions stay native 0..170 with no label LUT.
Wall-clock times from this repair are not E4 efficiency measurements.
"""
import json,hashlib,os,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from e4_special_backends import FreeDA
from e4_full_common import save,sha,utc,quality
R=Path('/data/tianyi/TF-OVOS');OUT=R/'runs/rebuttal/closure_20261011/freeda_full';OUT.mkdir(parents=True,exist_ok=True);(OUT/'predictions').mkdir(exist_ok=True)
rows=[json.loads(x) for x in (R/'data/manifests/coco_stuff171_val.jsonl').read_text().splitlines()]
for row in rows:
 for key in ['image_path','mask_path']:
  row[key]=str((R/'data/manifests'/row[key]).resolve()) if not Path(row[key]).is_absolute() else row[key]
journal=OUT/'per_image.jsonl';done=[]
if journal.exists():
 for l in journal.read_text().splitlines():
  if l.strip():done.append(json.loads(l))
for i,d in enumerate(done):
 assert d['image_id']==rows[i]['image_id'] and sha(OUT/d['prediction_file'])==d['prediction_sha256']
if len(done)==len(rows):print('All native predictions already complete',flush=True)
else:
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 model=FreeDA('freeda','coco_stuff171',OUT)
 assert len(model.meta['class_names'])==171 and not model.model.with_bg
 save(OUT/'protocol.json',dict(status='native_repair',images=5000,settings=model.meta,torch=torch.__version__,cudnn_tf32=True,matmul_tf32=False,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),R/'scripts/e4_special_backends.py']},prediction_encoding='native 0..170, uint8; no remapping, rejection, GT masking or shift',timing_scope='repair execution, not E4'))
 begin=time.time();start=len(done)
 for i,row in enumerate(rows[start:],start):
  with Image.open(row['image_path']) as im:rgb=im.convert('RGB')
  with torch.inference_mode():pred=model.infer(row,rgb).astype(np.int32)
  assert pred.min()>=0 and pred.max()<171
  path=OUT/'predictions'/(row['image_id']+'.png');tmp=path.with_suffix('.tmp.png');Image.fromarray(pred.astype(np.uint8)).save(tmp);tmp.replace(path)
  oldpath=Path('/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/coco_stuff164k')/path.name;old=np.array(Image.open(oldpath));keep=old!=255
  assert old.shape==pred.shape
  rec=dict(image_id=row['image_id'],prediction_file=str(path.relative_to(OUT)),prediction_sha256=sha(path),legacy_sha256=sha(oldpath),image_sha256=sha(row['image_path']),gt_sha256=sha(row['mask_path']),old_255_pixels=int((~keep).sum()),changed_non255_pixels=int((pred[keep]!=old[keep]).sum()),pixels=pred.size)
  with journal.open('a') as f:f.write(json.dumps(rec)+'\n');f.flush();os.fsync(f.fileno())
  if (i+1)%50==0 or i+1==len(rows):
   state=dict(processed=i+1,expected=5000,mean_seconds=(time.time()-begin)/(i+1-start),updated_utc=utc());save(OUT/'progress.json',state);print(state,flush=True)
 save(OUT/'inference_complete.json',dict(status='complete',images=5000,updated_utc=utc()))
# Independently read the saved PNGs and check their hashes; no live model output used.
conf=np.zeros((171,172),dtype=np.int64);joint=np.zeros((171,256),dtype=np.int64);inter=np.zeros_like(conf);records=[json.loads(x) for x in journal.read_text().splitlines()]
for i,(row,rec) in enumerate(zip(rows,records)):
 path=OUT/rec['prediction_file'];assert sha(path)==rec['prediction_sha256'];assert sha(row['mask_path'])==rec['gt_sha256'];pred=np.asarray(Image.open(path)).astype(np.int64);gt=np.asarray(Image.open(row['mask_path'])).astype(np.int64)
 old=np.asarray(Image.open(Path('/data/tianyi/TF-OVCOS/runs/artifacts/official_predictions/freeda/coco_stuff164k')/path.name)).astype(np.int64)
 valid=gt<171;cm=np.bincount(gt[valid]*172+pred[valid],minlength=171*172).reshape(171,172);conf+=cm
 joint+=np.bincount((pred*256+old).ravel(),minlength=171*256).reshape(171,256)
 if np.any(gt[valid]<80) and np.any(gt[valid]>=80):inter+=cm
np.save(OUT/'confusion.npy',conf);np.save(OUT/'intersection_confusion.npy',inter);np.save(OUT/'native_vs_archive_counts.npy',joint)
q=quality(conf);save(OUT/'verification.json',dict(status='complete',images=5000,independent_saved_png_rescore=True,changed_non255_pixels=sum(r['changed_non255_pixels'] for r in records),old_255_pixels=sum(r['old_255_pixels'] for r in records),**q,submitted_miou_percent=28.80,reproduces_submitted_to_two_decimals=round(q['miou_percent'],2)==28.8,updated_utc=utc()))
print(q['miou_percent'],flush=True)
