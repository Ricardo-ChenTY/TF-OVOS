"""Recompute only DINO guidance, retaining audited SAM masks and naming features.

New predictions/metrics are isolated from every frozen original experiment.
Legacy replay is checked against every original PNG before admitting a result.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
import torch
from tf_ovos.adapters.dinov2_sam import _Dinov2GuidanceMixin
from tf_ovos.dino_geometry import mask_grid_source_indices
from tf_ovos.metrics import dense_region_statistics

BASE=ROOT/'runs/rebuttal/closure_20261011/dino_geometry'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def loadrows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
class Guidance(_Dinov2GuidanceMixin):
 def __init__(self):
  self.torch=torch;self.device='cuda';self._init_dinov2_config();self._load_dinov2()

def score_both(adapter,rgb,packed,shape):
 inputs=adapter.dinov2_processor(images=rgb,return_tensors='pt');inputs={k:v.cuda() for k,v in inputs.items()}
 with torch.inference_mode():tokens=adapter.dinov2_model(**inputs).last_hidden_state[:,1:,:].float()[0]
 tokens=tokens/tokens.norm(dim=-1,keepdim=True).clamp_min(1e-6)
 gh,gw=adapter._infer_patch_grid(len(tokens),inputs);tokens=tokens[:gh*gw].reshape(gh,gw,-1)
 glob=tokens.reshape(-1,tokens.shape[-1]).mean(0);glob=glob/glob.norm().clamp_min(1e-6)
 idx=mask_grid_source_indices(shape,adapter.dinov2_processor,(gw,gh));valid=idx>=0
 raw=[[],[]]
 for p in packed:
  mask=np.unpackbits(p,count=int(np.prod(shape))).reshape(shape).astype(bool)
  legacy=np.asarray(Image.fromarray(mask.astype(np.uint8)*255).resize((gw,gh),Image.Resampling.NEAREST))>0
  aligned=np.zeros(idx.shape,dtype=bool);aligned[valid]=mask.ravel()[idx[valid]]
  for dest,selected in zip(raw,[legacy,aligned]):
   if not selected.any():dest.append(0.);continue
   feat=tokens[torch.as_tensor(selected,device='cuda')].mean(0);feat=feat/feat.norm().clamp_min(1e-6)
   distinct=float((1.-torch.dot(feat,glob)).clamp(min=0).item());area=float(selected.mean());prior=np.sqrt(max(area,1e-6))*(1.-np.sqrt(min(area,1.)))
   dest.append(distinct*(.5+prior))
 out=[]
 for values in raw:
  a=np.asarray(values,dtype=np.float32)
  if len(a) and a.max()>a.min():a=(a-a.min())/(a.max()-a.min())
  else:a=np.zeros_like(a)
  out.append(a)
 return out

def owner_map(packed,shape,sam,dino,weight):
 owner=np.full(shape,-1,dtype=np.int32);confidence=np.full(shape,-1,dtype=np.float32)
 for i,p in enumerate(packed):
  mask=np.unpackbits(p,count=int(np.prod(shape))).reshape(shape).astype(bool)
  score=float(sam[i])*(1.+weight*float(dino[i]));update=mask&(score>confidence);owner[update]=i;confidence[update]=score
 if not len(packed):owner[:]=0
 return owner

def metrics(conf,counts):
 n=conf.shape[0];tp=np.diag(conf[:,:n]);union=conf.sum(1)+conf[:,:n].sum(0)-tp
 per=np.divide(tp,union,out=np.full(n,np.nan),where=union>0)
 N,L5,W5,L75,W75=map(int,counts)
 return dict(miou=float(np.nanmean(per)),per_class_iou=per.tolist(),gt_class_regions=N,localized_pairs_at_05=L5,mismatch_pairs_at_05=W5,localized_pairs_at_075=L75,mismatch_pairs_at_075=W75,coverage_at_05=L5/N,mcmr_at_05=W5/L5 if L5 else None,ziou_gt_supported=float((tp[conf.sum(1)>0]==0).mean()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);a=ap.parse_args();out=BASE/('pilot' if a.limit else 'full');out.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
 jobs=[j for j in json.loads((ROOT/'runs/rebuttal/execution_20261007/jobs.json').read_text()) if j['method'].startswith('dinov2_sam') and j['seed']==0]
 adapter=Guidance();cache=BASE/'score_cache';cache.mkdir(parents=True,exist_ok=True)
 weights={str(p):sha(p) for p in Path(adapter.dinov2_cache_dir).rglob('*') if p.is_file() and p.suffix in ['.bin','.safetensors']}
 identity=dict(model=adapter.dinov2_model_id,weights=weights,processor=adapter.dinov2_processor.to_dict(),source_sha256={str(p):sha(p) for p in [Path(__file__),ROOT/'src/tf_ovos/dino_geometry.py']},torch=torch.__version__,tf32=False,cudnn_tf32=True)
 ident=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest();save(out/'protocol.json',dict(identity=identity,scope='full' if not a.limit else 'pilot',limit=a.limit,change='SAM mask projected through identical DINO resize and center crop before token selection',retained='SAM masks, naming crop features/text, mask ordering, SAM confidences, DINO weights, uncovered class0 rule; original outputs immutable'))
 summaries=[]
 for job in jobs:
  dest=out/job['name'];dest.mkdir(exist_ok=True)
  if (dest/'result.json').exists():summaries.extend(json.loads((dest/'result.json').read_text())['rows']);continue
  orig=Path(job['out']);config=json.loads((orig/'resolved_configuration.json').read_text());vc={k:v for k,v in config.items() if k not in {'text_features_sha256','num_classes','task','prompt_template'}};fingerprint=hashlib.sha256(json.dumps(vc,sort_keys=True).encode()).hexdigest()
  rows=loadrows(orig/'manifest.jsonl');inputs=json.loads((orig/'input_hashes.json').read_text());levels=job['sizes'];n=int(json.loads((orig/'plan.json').read_text())['num_eval_classes']);text=torch.from_numpy(np.load(orig/'text_features.npy')).cuda()
  if a.limit:rows=rows[:a.limit]
  hist={k:np.zeros((n,k),dtype=np.int64) for k in levels};counts={k:np.zeros(5,dtype=np.int64) for k in levels};intersection={k:np.zeros((n,k),dtype=np.int64) for k in levels};changed={k:[0,0] for k in levels};max_score_diff=0.;start=time.time();records=[]
  for k in levels:(dest/f'candidates_{k}').mkdir(exist_ok=True)
  for i,row in enumerate(rows):
   inp=inputs[i];assert sha(row['image_path'])==inp['image_sha256'];assert sha(row['mask_path'])==inp['gt_sha256']
   key=hashlib.sha256((fingerprint+inp['image_sha256']).encode()).hexdigest();cp=Path(job['cache_dir'])/(key+'.npz')
   with np.load(cp) as z:regions={k:z[k] for k in z.files}
   shape=tuple(map(int,regions['shape']));packed=regions['packed_masks'];oldowner=regions['owner'];oldscore=regions['dino_scores']
   ck=hashlib.sha256((ident+inp['image_sha256']+str(shape)).encode()+packed.tobytes()).hexdigest();scorepath=cache/(ck+'.npz')
   if scorepath.exists():
    with np.load(scorepath) as z:legacy=z['legacy'];aligned=z['aligned']
   else:
    with Image.open(row['image_path']) as im:legacy,aligned=score_both(adapter,im.convert('RGB'),packed,shape)
    np.savez_compressed(scorepath,legacy=legacy,aligned=aligned)
   diff=float(np.abs(legacy-oldscore).max()) if len(oldscore) else 0.;max_score_diff=max(max_score_diff,diff)
   if diff>2e-5:raise RuntimeError(('Legacy DINO score reproduction failed',job['name'],row['image_id'],diff))
   newowner=owner_map(packed,shape,regions['sam_confidences'],aligned,config['dinov2_weight'])
   gt=np.asarray(Image.open(row['mask_path'])).astype(np.int64);assert gt.shape==shape
   features=torch.from_numpy(regions['features']).cuda().to(text.dtype);perrecord=dict(image_id=row['image_id'],cache_sha256=sha(cp),score_error=diff,score_file=str(scorepath),predictions={})
   for k in levels:
    with torch.inference_mode():best=(features@text[:k].T).argmax(1).cpu().numpy()
    old=np.zeros(shape,dtype=np.uint16);old[oldowner>=0]=best[oldowner[oldowner>=0]]
    oldpath=orig/f'candidates_{k}'/f'{i:06d}.png';stored=np.asarray(Image.open(oldpath));assert np.array_equal(old,stored),('Legacy replay mismatch',job['name'],i,k)
    new=np.zeros(shape,dtype=np.uint16);new[newowner>=0]=best[newowner[newowner>=0]]
    changed[k][0]+=int((new!=old).sum());changed[k][1]+=new.size
    p=dest/f'candidates_{k}'/f'{i:06d}.png';Image.fromarray(new).save(p)
    # Verify saved labels before scoring, rather than score an unexported array.
    saved=np.asarray(Image.open(p));assert np.array_equal(saved,new)
    st=dense_region_statistics(saved,gt,n,job.get('void_label',255),num_candidates=k);hist[k]+=st.confusion
    c5,correct5=st.counts(.5);c75,correct75=st.counts(.75);counts[k]+=np.array([len(st.gt_ids),c5,c5-correct5,c75,c75-correct75])
    if job['dataset']=='coco_stuff171' and np.any(st.gt_ids<80) and np.any(st.gt_ids>=80):intersection[k]+=st.confusion
    perrecord['predictions'][str(k)]=dict(sha256=sha(p),legacy_sha256=sha(oldpath))
   records.append(perrecord)
   if (i+1)%50==0 or i+1==len(rows):
    state=dict(job=job['name'],processed=i+1,expected=len(rows),seconds_per_image=(time.time()-start)/(i+1),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save(out/'progress.json',state);print(state,flush=True)
  result=[]
  for k in levels:
   m=metrics(hist[k],counts[k]);m.update(images=len(rows),scope='full' if not a.limit else 'pilot',geometry='DINO_RGB_aligned_mask_resize_center_crop',legacy_replay_images=len(rows),max_legacy_score_error=max_score_diff,changed_pixels=changed[k][0],pixels=changed[k][1]);save(dest/f'candidates_{k}/metrics.json',m);np.save(dest/f'candidates_{k}/confusion.npy',hist[k]);np.save(dest/f'candidates_{k}/intersection_confusion.npy',intersection[k])
   oldm=json.loads((orig/f'candidates_{k}/metrics.json').read_text());r=dict(method=job['method'],dataset=job['dataset'],candidates=k,images=len(rows),corrected_miou_percent=100*m['miou'],old_full_miou_percent=100*oldm['miou'],change_pp=100*(m['miou']-oldm['miou']) if not a.limit else None,changed_pixels=changed[k][0],pixels=changed[k][1],source=str(dest/f'candidates_{k}/metrics.json'));result.append(r)
  save(dest/'replay_verification.json',dict(images=len(rows),all_old_pngs_exact=True,max_legacy_dino_score_error=max_score_diff,records=records));save(dest/'result.json',dict(status='complete',rows=result));summaries.extend(result);save(out/'summary.json',summaries)
 save(out/'complete.json',dict(status='complete',jobs=len(jobs),metric_rows=len(summaries),scope='full' if not a.limit else 'pilot',updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
if __name__=='__main__':main()
