"""Reconcile changed implementations into one seed-0 report; keep frozen history."""
import csv,json,sys,time,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import numpy as np
from PIL import Image
from tf_ovos.metrics import dense_region_statistics
from report_fixed_gt_seed0 import endpoints,nonmonotonic,CURVE_FIELDS,END_FIELDS
B=R/'runs/rebuttal/closure_20261011';OLD=R/'docs/rebuttal/evidence/20261011_final';OUT=B/'unified'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def rows(p):return list(csv.DictReader(Path(p).open()))
def csvout(p,rr,fields=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rr[0]));w.writeheader();w.writerows(rr)
def coco(method,job):
 out=B/'coco_diagnostics'/method
 if (out/'result.json').exists():return read(out/'result.json')['rows']
 manifest=[json.loads(x) for x in (Path(job['out'])/'manifest.jsonl').read_text().splitlines()];full=np.zeros((171,171),dtype=np.int64);inter=np.zeros_like(full);counts=np.zeros((2,3),dtype=np.int64);images=[];hashes={}
 if method=='freeda':ref=B/'freeda_full';source=ref/'intersection_confusion.npy'
 else:ref=B/'dino_geometry/full'/job['name']/'candidates_171';source=ref/'intersection_confusion.npy'
 for i,r in enumerate(manifest):
  p=ref/'predictions'/(r['image_id']+'.png') if method=='freeda' else ref/('%06d.png'%i)
  gt=np.asarray(Image.open(r['mask_path'])).astype(np.int64);pred=np.asarray(Image.open(p)).astype(np.int64);st=dense_region_statistics(pred,gt,171);full+=st.confusion
  if not (np.any(st.gt_ids<80) and np.any(st.gt_ids>=80)):continue
  images.append(r['image_id']);inter+=st.confusion;hashes[str(p)]=sha(p)
  for gi,selected in enumerate([st.gt_ids<80,st.gt_ids>=80]):
   loc=selected&(st.best_ious>=.5);counts[gi]+=np.array([int(selected.sum()),int(loc.sum()),int((loc&(st.best_ids!=st.gt_ids)).sum())])
  if (i+1)%1000==0:print('COCO rescore',method,i+1,flush=True)
 assert len(images)==4894;prior=read(R/'runs/rebuttal/final_analysis_20261009/coco_intersection_provenance.json');assert images==prior['included_image_ids']
 expected=np.load(source);assert np.array_equal(inter,expected[:,:171]);assert not expected[:,171:].any()
 expectedfull=np.load(ref/'confusion.npy');assert np.array_equal(full,expectedfull[:,:171]);assert not expectedfull[:,171:].any()
 tp=np.diag(inter);iou=tp/(inter.sum(1)+inter.sum(0)-tp);rr=[]
 for gi,(group,sl) in enumerate([('thing',slice(0,80)),('stuff',slice(80,171))]):
  n,l,w=map(int,counts[gi]);assert n==[14134,27633][gi]
  rr.append(dict(method=method,group=group,images=4894,candidates=171,miou_percent=100*float(iou[sl].mean()),overall_miou_percent=100*float(iou.mean()),gt_pairs=n,localized_correct=l-w,localized_wrong=w,unlocalized=n-l,coverage=l/n,mcmr=w/l,evidence='native_label_restore' if method=='freeda' else 'DINO_mask_geometry_corrected',source_confusion=str(source)))
 save(out/'result.json',dict(status='complete',rows=rr,images=images,prediction_sha256=hashes,independent_confusion_matches=True));return rr

def main():
 assert read(B/'dino_geometry/full/complete.json')['jobs']==12
 assert read(B/'freeda_full/verification.json')['status']=='complete'
 OUT.mkdir(parents=True,exist_ok=True);summ=read(B/'dino_geometry/full/summary.json');index={(r['method'],r['dataset'],r['candidates']):r for r in summ};assert len(index)==24
 curves=rows(OLD/'fixed_gt/curves.csv')
 for r in curves:
  for k in ['candidate_seed','candidate_draws','images','base_candidates','candidates']:r[k]=int(r[k])
  r['miou_percent']=float(r['miou_percent']);k=(r['method'],r['dataset'],r['candidates'])
  if k in index:r.update(miou_percent=index[k]['corrected_miou_percent'],source=index[k]['source'],source_sha256=sha(index[k]['source']),geometry='original_frame_with_DINO_aligned_center_crop_guidance')
 assert len(curves)==136;ends=endpoints(curves);steps=nonmonotonic(curves);assert len(ends)==34
 csvout(OUT/'fixed_gt/curves.csv',curves,CURVE_FIELDS);csvout(OUT/'fixed_gt/endpoints.csv',ends,END_FIELDS);csvout(OUT/'fixed_gt/nonmonotonic_steps.csv',steps,['method','dataset','candidate_seed','from_candidates','to_candidates','increase_pp'])
 common=rows(OLD/'diagnostics/common_gt_counts.csv')
 for r in common:
  k=(r['method'],r['dataset'],459 if r['dataset']=='context459' else 847)
  if k not in index:continue
  src=index[k]['source'];m=read(src);suffix='05' if float(r['iou_threshold'])==.5 else '075';n=m['gt_class_regions'];l=m['localized_pairs_at_'+suffix];w=m['mismatch_pairs_at_'+suffix]
  r.update(gt_pairs=n,localized_correct=l-w,localized_wrong=w,unlocalized=n-l,correct_fraction=(l-w)/n,wrong_fraction=w/n,unlocalized_fraction=(n-l)/n,coverage=l/n,mcmr=w/l,source=src,source_sha256=sha(src),evidence='geometry_corrected_seed0_full_predictions')
 for r in common:
  assert int(r['gt_pairs'])==int(r['localized_correct'])+int(r['localized_wrong'])+int(r['unlocalized'])
  assert int(r['gt_pairs'])=={'context459':33026,'ade847':20369}[r['dataset']]
 assert len(common)==56 and len({(r['method'],r['dataset'],float(r['iou_threshold'])) for r in common})==56
 csvout(OUT/'diagnostics/common_gt_counts.csv',common)
 jobs={j['method']:j for j in read(R/'runs/rebuttal/execution_20261007/jobs.json') if j['dataset']=='coco_stuff171' and j['seed']==0}
 updated=[]
 for method in ['dinov2_sam_clip','dinov2_sam_siglip','freeda']:updated+=coco(method,jobs.get(method,jobs['dinov2_sam_clip']))
 coco_rows=rows(OLD/'diagnostics/coco_thing_stuff_intersection.csv');cocoindex={(r['method'],r['group']):r for r in updated};coco_rows=[cocoindex.get((r['method'],r['group']),r) for r in coco_rows];csvout(OUT/'diagnostics/coco_thing_stuff_intersection.csv',coco_rows)
 comparison=rows(OLD/'diagnostics/submitted_comparison.csv');print('Comparison fields',list(comparison[0]),flush=True)
 for r in comparison:
  k=next((k for k in index if k[0]==r['method'] and k[1]==r['dataset'] and k[2]=={'voc20':20,'context59':59,'context459':459,'ade150':150,'ade847':847,'coco_stuff171':171}[r['dataset']]),None)
  if k:
   rr=index[k];r['new_miou']=rr['corrected_miou_percent'];r['change_pp']=rr['corrected_miou_percent']-float(r['submitted_miou']);r['source']=rr['source']
 csvout(OUT/'diagnostics/submitted_comparison.csv',comparison)
 compact=[]
 for method in ['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip']:
  scores={ds:float(next(r['new_miou'] for r in comparison if r['method']==method and r['dataset']==ds)) for ds in ['voc20','context59','ade150','coco_stuff171']}
  compact.append(dict(method=method,**scores,mean_miou_percent=sum(scores.values())/4,worst_miou_percent=min(scores.values()),worst_dataset=min(scores,key=scores.get),scope='four compact native targets; candidate seed0; DINO geometry corrected'))
 csvout(OUT/'diagnostics/compact_mean_worst.csv',compact)


 # Updated COCO plot and selection provenance must match the corrected table.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 provenance=read(R/'runs/rebuttal/final_analysis_20261009/coco_intersection_provenance.json')
 provenance.update(current_version='20261011_geometry_and_FreeDA_label_restore',source_sha256={r['source_confusion']:sha(r['source_confusion']) for r in coco_rows},script_sha256=sha(__file__),updated_method_diagnostics={m:str(B/'coco_diagnostics'/m/'result.json') for m in ['freeda','dinov2_sam_clip','dinov2_sam_siglip']})
 save(OUT/'diagnostics/coco_intersection_provenance.json',provenance)
 methods=list(dict.fromkeys(r['method'] for r in coco_rows));xx=np.arange(len(methods));fig,ax=plt.subplots(figsize=(11,5))
 for off,group,color in [(-.2,'thing','#277da1'),(.2,'stuff','#e09136')]:
  values=[float(next(r['miou_percent'] for r in coco_rows if r['method']==m and r['group']==group)) for m in methods]
  bars=ax.bar(xx+off,values,.4,label=group,color=color);ax.bar_label(bars,fmt='%.1f',fontsize=8)
 ax.set_xticks(xx,methods,rotation=25,ha='right');ax.set(ylabel='mIoU (%)',title='COCO: same 4,894 images; corrected FreeDA labels and DINO geometry');ax.legend();fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(OUT/('diagnostics/coco_thing_stuff_intersection.'+ext),dpi=180)
 plt.close(fig)
 # Compare native saved labels to the old timed FreeDA COCO array hashes.
 timing=[r for r in [json.loads(x) for x in (R/'runs/rebuttal/e4_20261009/freeda/per_image.jsonl').read_text().splitlines()] if r['dataset']=='coco_stuff171'];paired=[]
 for r in timing:
  p=B/'freeda_full/predictions'/(r['image_id']+'.png');a=np.asarray(Image.open(p)).astype(np.int64);digest=hashlib.sha256(a.tobytes()).hexdigest();paired.append(dict(image_id=r['image_id'],matches_native_int64_array=digest==r['prediction_sha256'],new_array_sha256=digest,old_array_sha256=r['prediction_sha256']))
 save(B/'freeda_e4_pairing.json',dict(compared=len(paired),exact_matches=sum(r['matches_native_int64_array'] for r in paired),rows=paired,scope='COCO fixed25-image timed subset only. Old timing uses cudnn TF32-off; restored quality uses native cudnn TF32-on; do not assume untested images identical.'))
 csvout(OUT/'dino_geometry_changes.csv',summ)
 audit=read(B/'freeda_label_corruption_audit.json');f=read(B/'freeda_full/verification.json');joint=np.load(B/'freeda_full/native_vs_archive_counts.npy');erased=np.flatnonzero(joint[:,255]).tolist();mismatches=int(joint[:,:171].sum()-np.trace(joint[:,:171]));unexpected=int(joint[:,171:255].sum())
 audit.update(status='full_verified',restored_miou_percent=f['miou_percent'],full_images=5000,full_changed_non255_pixels=mismatches,full_old_255_pixels=int(joint[:,255].sum()),erased_native_classes=erased,unexpected_archive_labels=unexpected,reproduces_submitted_to_two_decimals=f['reproduces_submitted_to_two_decimals'],causal_limit='Full native rerun establishes lost legitimate class labels in archived PNGs. The exact historical destructive conversion command is not recovered; its category-ID offset origin is inferred from the exact hole signature.')
 assert erased==audit['missing_native_classes'] and unexpected==0 and mismatches==f['changed_non255_pixels'];save(B/'freeda_label_corruption_audit.json',audit)
 save(OUT/'verification.json',dict(status='complete',updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),curve_rows=len(curves),endpoint_rows=len(ends),descending_endpoints=sum(r['drop_pp']>0 for r in ends),nonmonotonic_steps=len(steps),common_gt_rows=len(common),coco_rows=len(coco_rows),coco_images=4894,stuff_lower_methods=sum(float(next(r['miou_percent'] for r in coco_rows if r['method']==m and r['group']=='stuff'))<float(next(r['miou_percent'] for r in coco_rows if r['method']==m and r['group']=='thing')) for m in {r['method'] for r in coco_rows}),freeda=f,source_sha256={r['source']:r['source_sha256'] for r in curves},scope='Single candidate seed 0. Replaces DINO results in all main tables and FreeDA COCO results. Historical multi-seed legacy DINO outputs remain historical, not aligned implementation evidence.'))
 print('UNIFIED COMPLETE',flush=True)
if __name__=='__main__':
 if '--dino-coco-only' in sys.argv:
  jobs={j['method']:j for j in read(R/'runs/rebuttal/execution_20261007/jobs.json') if j['dataset']=='coco_stuff171' and j['seed']==0}
  for method in ['dinov2_sam_clip','dinov2_sam_siglip']:
   job=jobs[method]
   while not (B/'dino_geometry/full'/job['name']/'result.json').exists():time.sleep(10)
   coco(method,job)
 else:main()

