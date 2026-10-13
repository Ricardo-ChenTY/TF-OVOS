#!/usr/bin/env python3
"""Verify every paired prediction and publish the isolated VOC correction evidence."""
import csv,hashlib,json,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from e4_full_common import confusion_matrix,unpack_confusion,quality
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/rebuttal/resclip_voc_release_20261013'
PUBLIC=ROOT/'docs/rebuttal/evidence/resclip_voc_release_20261013'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[json.loads(s) for s in (OUT/'manifests/voc20.jsonl').read_text().splitlines()]
 runs={'same_runtime_slide224':OUT/'baseline_replay','release_slide0':OUT/'jobs/resclip/voc20'}
 checks=[];results={};saved={}
 for label,folder in runs.items():
  result=json.loads((folder/'result.json').read_text());assert result['status']=='complete' and result['images']==1449
  db=sqlite3.connect('file:'+str(folder/'measurements.sqlite')+'?mode=ro',uri=True);cm=np.zeros((20,21),np.int64);predictions=[];metadata=[]
  for image_id,index,record,blob in db.execute('select * from images order by ordinal'):
   row=rows[index];r=json.loads(record);assert image_id==row['image_id'];p=folder/r['prediction_file'];assert sha(p)==r['prediction_file_sha256']
   gt=np.asarray(Image.open(row['mask_path']));a=np.asarray(Image.open(p));assert a.min()>=0 and a.max()<20
   count=confusion_matrix(gt,a,20);assert np.array_equal(count,unpack_confusion(blob,20));cm+=count
   predictions.append(a);metadata.append(dict(image_id=image_id,prediction_sha256=sha(p),gt_sha256=row['mask_sha256']))
  db.close();assert len(predictions)==1449 and np.array_equal(cm,np.load(folder/'confusion.npy'))
  q=quality(cm);assert abs(q['miou_percent']-result['miou_percent'])<1e-10
  results[label]=dict(images=1449,miou_percent=q['miou_percent'],online_mean_seconds=result['online_mean_seconds'],online_total_seconds=result['online_total_seconds'],setup_seconds_by_session=[json.loads(x.read_text())['setup_seconds'] for x in folder.glob('session_*.json')],source_result_sha256=sha(folder/'result.json'))
  saved[label]=predictions;checks.append(dict(variant=label,images=1449,confusion_recomputed_from_png=True,records=metadata))
 old=json.loads((OUT/'historical_rescore.json').read_text());a,b=list(runs)
 changed=sum(int((x!=y).sum()) for x,y in zip(saved[a],saved[b]));pixels=sum(x.size for x in saved[a])
 settings=[json.loads((f/'metadata.json').read_text())['settings']['model'] for f in runs.values()]
 differences={k:[settings[0].get(k),settings[1].get(k)] for k in set(settings[0])|set(settings[1]) if settings[0].get(k)!=settings[1].get(k)}
 assert differences=={'slide_crop':[224,0]},differences
 report=dict(status='verified',images_per_variant=1449,changed_model_parameters=differences,
             historical_submitted_miou=78.51,historical_png_rescored_miou=old['miou_percent'],
             same_runtime_control=results[a],release_correction=results[b],
             paired_slide_crop_effect_pp=results[b]['miou_percent']-results[a]['miou_percent'],
             historical_to_current_control_pp=results[a]['miou_percent']-old['miou_percent'],
             corrected_vs_historical_pp=results[b]['miou_percent']-old['miou_percent'],
             changed_prediction_pixels=changed,total_prediction_pixels=pixels,
             scope='VOC only; official slide_crop restored, benchmark class names retained. Paired effect isolates window setting in same current runtime; historic comparison also includes runtime differences.',
             not_changed='Other target scores, large-vocabulary adaptation, fixed-GT experiments; no full unified-recipe experiment',
             preprocessing=json.loads((runs[b]/'metadata.json').read_text())['settings']['test_pipeline'])
 PUBLIC.mkdir(parents=True,exist_ok=True)
 (PUBLIC/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 (PUBLIC/'prediction_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
 entries=[dict(variant='historical_saved_png',images=1449,miou_percent=old['miou_percent'],online_mean_seconds='')]+[dict(variant=k,**{f:v[f] for f in ['images','miou_percent','online_mean_seconds']}) for k,v in results.items()]
 with (PUBLIC/'comparison.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
 base=json.loads((ROOT/'configs/rebuttal/submitted_pdf_baseline.json').read_text());base=next(r for r in base['rows'] if r['method']=='resclip')
 fields=['voc20','context59','ade150','coco_stuff171'];scores={k:base[k] for k in fields};scores['voc20']=results[b]['miou_percent']
 derived=dict(**scores,mean=sum(scores.values())/4,worst=min(scores.values()),scope='VOC full precision; other three cells retained from rounded submitted values. Update final report card from full-precision sources when available.')
 (PUBLIC/'compact_summary_update.json').write_text(json.dumps(derived,indent=2)+'\n')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
