"""Compare archived qualitative cases to current seed-0 SAM+SigLIP maps; CPU only."""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1]
OLD=Path('/data/tianyi/TF-OVCOS')
OUT=R/'runs/rebuttal/figure_audit_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 OUT.mkdir(exist_ok=True)
 cases=json.loads((OLD/'runs/analysis/qual_main_four_events_fixed_cases.json').read_text())[4:]
 jobs=json.loads((R/'runs/rebuttal/execution_20261007/jobs.json').read_text());records=[]
 for c in cases:
  ds={'ade20k':'ade150'}.get(c['dataset'],c['dataset'])
  job=next(j for j in jobs if j['dataset']==ds and j['method']=='sam_amg_siglip' and j['seed']==0)
  olddir=OLD/'runs/sam_amg_siglip'/c['manifest'];oldrow=next(r for r in readrows(olddir/'predictions.jsonl') if r['image_id']==c['image_id']);oldpath=olddir/oldrow['mask_path']
  folder=Path(job['out']);i,row=next((i,r) for i,r in enumerate(readrows(folder/'manifest.jsonl')) if Path(r['image_id']).name==Path(c['image_id']).name)
  newpath=folder/f"candidates_{job['sizes'][0]}"/f'{i:06d}.png'
  old=np.asarray(Image.open(oldpath)).astype(int);new=np.asarray(Image.open(newpath)).astype(int);gt=np.asarray(Image.open(row['mask_path']));target=gt==c['target_label'];vocab=Path(job['base_vocab']).read_text().splitlines()
  assert old.shape==new.shape==gt.shape and target.any()
  details=[];masks=[]
  for a in [old,new]:
   label=int(np.bincount(a[target]).argmax());mask=a==label;masks.append(mask)
   details.append(dict(predicted_label_id=label,predicted_name=vocab[label],mask_pixels=int(mask.sum()),target_overlap_pixels=int((mask&target).sum()),target_iou=float((mask&target).sum()/(mask|target).sum())))
  record=dict(image_id=c['image_id'],dataset=ds,target=c['target_name'],target_pixels=int(target.sum()),image_pixels=old.size,
              changed_class_pixels=int((old!=new).sum()),changed_class_percent=100*float((old!=new).mean()),
              selected_mask_changed_pixels=int((masks[0]!=masks[1]).sum()),old=details[0],current=details[1],
              paths=dict(old_prediction=str(oldpath),new_prediction=str(newpath),gt=row['mask_path']),
              sha256={str(p):sha(p) for p in [oldpath,newpath,Path(row['mask_path']),Path(job['base_vocab'])]})
  # Existing archived full-scene tiles can be identified without relying on captions:
  # changed RGB pixels vs their input tile exactly equal the old selected-class mask.
  if c['target_name'] in ['signboard','bus']:
   stem='case_study_mismatch_signboard' if c['target_name']=='signboard' else 'case_study_small_object_bus'
   fig=OLD/'runs/analysis'/(stem+'.png');a=np.asarray(Image.open(fig).convert('RGB'))
   displayed=np.any(a[71:247,1034:1269]!=a[71:247,82:317],axis=2)
   rendered=np.asarray(Image.fromarray(masks[0]).resize((235,176),Image.Resampling.NEAREST))
   assert np.array_equal(displayed,rendered)
   record['archived_panel_verified']=dict(path=str(fig),sha256=sha(fig),full_scene_selected_mask_matches_exactly=True,displayed_mask_pixels=int(displayed.sum()),scope='archived PNG full-scene SAM+SigLIP tile; final appendix PDF unavailable')
  records.append(record)
 result=dict(status='complete',scope='Four historical case IDs; submitted 15-page pdf.pdf has no Figures 6-8; final appendix correspondence remains unverified',new_gpu_runs=0,records=records)
 (OUT/'prediction_comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 for r in records:print(r['target'],r['changed_class_percent'],r['selected_mask_changed_pixels'],r['old']['predicted_name'],r['current']['predicted_name'])
if __name__=='__main__':main()
