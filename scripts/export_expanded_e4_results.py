"""Verify and export completed expanded E4 evidence without changing quality."""
import csv,hashlib,json,shutil,tarfile
from pathlib import Path
import numpy as np
ROOT=Path('/data/tianyi/TF-OVOS')
RUNS=Path('/data/sdd/tianyi/TF-OVOS-rebuttal/runs')
P=RUNS/'proposal_e4_1000_per_target_20261015'
C=RUNS/'corrclip_e4_1000_per_target_20261015'
OUT=ROOT/'docs/rebuttal/evidence/e4_expanded_results_20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def csvrows(p):return list(csv.DictReader(p.open()))
def main():
 OUT.mkdir(parents=True,exist_ok=True);jobs=[];paths=[]
 for run in [P,C]:
  protocol=json.loads((run/'protocol.json').read_text())
  assert json.loads((run/'queue_status.json').read_text())['status']=='complete'
  checks=json.loads((run/'interval_verification.json').read_text())
  for path,digest in checks.get('source_sha256',checks.get('sources',{})).items():assert sha(run/path)==digest
  for method in protocol['methods']:
   for ds in protocol['datasets']:
    out=run/'jobs'/method/ds;r=json.loads((out/'result.json').read_text());a=json.loads((out/'prediction_pairing.json').read_text())
    records=[json.loads(x) for x in (out/'per_image.jsonl').read_text().splitlines()]
    manifest=[json.loads(x) for x in (run/'manifests'/(ds+'.jsonl')).read_text().splitlines()]
    assert r['images']==a['images']==a['exact_matches']==len(records)==1000
    assert all(x['changed_pixels']==0 for x in a['records'])
    assert [x['image_id'] for x in records]==[x['image_id'] for x in manifest]==[x['image_id'] for x in a['records']]
    assert len({x['image_id'] for x in records})==1000
    assert all(not x['observed_other_gpu_pids'] for x in records)
    mean=float(np.mean([x['online_seconds'] for x in records]));assert abs(mean-r['online_mean_seconds'])<1e-12
    jobs.append(dict(method=method,dataset=ds,images=1000,exact_matches=1000,online_mean_seconds=mean,finished_utc=r['completed_utc']))
    for name in ['result.json','metadata.json','per_image.jsonl','prediction_pairing.json']:
     paths.append((out/name,run.name+'/jobs/'+method+'/'+ds+'/'+name))
  for name in ['queue_status.json','progress.json','protocol.json','source_sha256.json','interval_verification.json','latency_ci.csv']:
   paths.append((run/name,run.name+'/'+name))
  for path in (run/'manifests').glob('*.jsonl'):paths.append((path,run.name+'/manifests/'+path.name))
 assert len(jobs)==20 and sum(x['images'] for x in jobs)==20000
 for ds in ['voc20','context59','ade150','coco_stuff171']:assert (P/'manifests'/(ds+'.jsonl')).read_bytes()==(C/'manifests'/(ds+'.jsonl')).read_bytes()
 for source,name in [(C/'latency_ci_five_configurations.csv','latency_ci_five_configurations.csv'),(P/'paired_dino_overhead_ci.csv','paired_dino_overhead_ci.csv')]:shutil.copy2(source,OUT/name)
 full=csvrows(ROOT/'docs/rebuttal/evidence/reply_support_20261012/derived/current_quality_latency_pairs.csv')
 qualities={r['method']:float(r['full_four_target_mean_miou_percent']) for r in full}
 combined=[]
 for row in csvrows(OUT/'latency_ci_five_configurations.csv'):
  if row['target']!='equal_target_mean':continue
  name=row['method'];key='corrclip_online' if name=='corrclip_online_sam2_8' else name
  mean=np.mean([j['online_mean_seconds'] for j in jobs if j['method']==key]);assert abs(mean-float(row['online_mean_seconds']))<1e-12
  combined.append(dict(**row,unchanged_full_quality_miou_percent=qualities[name],quality_scope='existing full native compact run; no substitution of sample mIoU'))
 with (OUT/'quality_latency_pairs.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(combined[0]));w.writeheader();w.writerows(combined)
 verification=dict(status='complete',completed_jobs=20,timed_predictions=20000,exact_prediction_matches=20000,images_per_target_per_config=1000,
  completed_utc=json.loads((C/'queue_status.json').read_text())['updated_utc'],jobs=jobs,
  checks=['Same image IDs/order across five configurations','All completed prediction-pairing records have zero changed pixels','Recomputed dataset and equal-target time means from per-image records','Statistical source hashes match frozen verification files','Full-quality scores retained unchanged'],
  limits=['Five configurations only, not all benchmark methods','Expanded sample, not full validation sets','Image-bootstrap intervals conditional on measured sessions','cuDNN TF32 on for four proposals and off for online CorrCLIP; matmul TF32 off for all','File I/O and initialization excluded from online timing'])
 (OUT/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
 art=ROOT/'docs/rebuttal/artifacts/e4_expanded_results_20261016.tar.gz'
 with tarfile.open(art,'w:gz') as t:
  for path,name in paths:t.add(path,arcname='e4_expanded_results_20261016/'+name)
  for path in OUT.iterdir():
   if path.name!='manifest.json':t.add(path,arcname='e4_expanded_results_20261016/summary/'+path.name)
 art.with_suffix(art.suffix+'.sha256').write_text(sha(art)+'  '+art.name+'\n')
 (OUT/'manifest.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in OUT.iterdir() if p.name!='manifest.json'},archive=str(art.relative_to(ROOT)),archive_sha256=sha(art)),indent=2)+'\n')
 print(json.dumps(dict(jobs=len(jobs),exact_matches=20000,archive_bytes=art.stat().st_size)))
if __name__=='__main__':main()
