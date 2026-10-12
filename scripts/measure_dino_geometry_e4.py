"""Uncached end-to-end timing paired to geometry-corrected full predictions."""
import argparse,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
import torch
import measure_e4 as engine
from e4_gpu_guard import GPUGuard
from e4_full_common import sha,save,utc,load_rows
engine.torch=torch
BASE=ROOT/'runs/rebuttal/closure_20261011'
def main():
 p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--dataset',required=True);a=p.parse_args();out=BASE/'dino_e4'/a.method/a.dataset;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():return
 torch.set_num_threads(4);torch.set_num_interop_threads(4);torch.manual_seed(0);np.random.seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
 manifests=ROOT/'runs/rebuttal/e4_targeted_20261009/proposal_same_recipe/manifests';rows=load_rows(manifests/(a.dataset+'.jsonl'));warm=load_rows(manifests/(a.dataset+'_warmup.jsonl'))
 jobs=json.loads((ROOT/'runs/rebuttal/execution_20261007/jobs.json').read_text());job=next(j for j in jobs if j['method']==a.method and j['dataset']==a.dataset and j['seed']==0)
 original=Path(job['out']);indices={r['image_id']:i for i,r in enumerate(load_rows(original/'manifest.jsonl'))};refdir=BASE/'dino_geometry/full'/job['name']/('candidates_'+str(job['sizes'][0]));assert (refdir/'metrics.json').exists()
 images={r['image_id']:Image.open(r['image_path']).convert('RGB') for r in warm+rows};guard=GPUGuard();guard.wait_idle();start=time.perf_counter();backend=engine.Local(a.method,a.dataset,out);torch.cuda.synchronize();setup=time.perf_counter()-start
 sources=[Path(__file__),ROOT/'scripts/measure_e4.py',ROOT/'src/tf_ovos/dino_geometry.py',ROOT/'src/tf_ovos/adapters/dinov2_sam.py',ROOT/'src/tf_ovos/adapters/sam_amg_clip.py',ROOT/'src/tf_ovos/adapters/sam_amg_siglip.py']
 save(out/'protocol.json',dict(method=a.method,dataset=a.dataset,settings=backend.meta,matmul_tf32=False,cudnn_tf32=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(),cpu_threads=4,warmup_images=len(warm),measured_images=len(rows),setup_seconds=setup,vision_cache=False,timing='decoded RGB through online SAM, DINO, naming and CPU label output; CUDA synchronized; file IO and initialization excluded',sources={str(x):sha(x) for x in sources},reference_metrics_sha256=sha(refdir/'metrics.json'),manifest_sha256=sha(manifests/(a.dataset+'.jsonl'))))
 records=[]
 with torch.inference_mode():
  for r in warm:guard.wait_idle();backend.infer(r,images[r['image_id']]);torch.cuda.synchronize()
  for r in rows:
   for attempt in range(20):
    guard.wait_idle();guard.begin();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();pred=backend.infer(r,images[r['image_id']]);torch.cuda.synchronize();elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated()/1024**2;others=guard.end()
    if not others:break
   else:raise RuntimeError('Repeated GPU contention')
   ref=refdir/('%06d.png'%indices[r['image_id']]);expected=np.asarray(Image.open(ref));assert pred.shape==expected.shape
   changed=int((pred!=expected).sum());assert changed==0,('Uncached prediction differs from corrected full cache replay',a.method,a.dataset,r['image_id'],changed)
   dest=out/(r['image_id']+'.png');Image.fromarray(pred.astype(np.uint16)).save(dest)
   rec=dict(image_id=r['image_id'],seconds=elapsed,peak_allocated_mib=peak,changed_pixels=changed,pixels=pred.size,prediction_sha256=sha(dest),reference_path=str(ref),reference_sha256=sha(ref),other_gpu_pids=others)
   records.append(rec);save(out/'progress.json',dict(processed=len(records),expected=len(rows),updated_utc=utc()))
   with (out/'per_image.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 guard.close();times=np.array([r['seconds'] for r in records]);save(out/'result.json',dict(status='complete',method=a.method,dataset=a.dataset,images=len(records),exact_matches=len(records),mean_seconds=float(times.mean()),p90_seconds=float(np.percentile(times,90)),peak_allocated_mib=max(r['peak_allocated_mib'] for r in records),completed_utc=utc(),records=records))
 print(a.method,a.dataset,'COMPLETE',float(times.mean()),flush=True)
if __name__=='__main__':main()
