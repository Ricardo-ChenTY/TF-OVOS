"""Freeze the corrected evidence without overwriting previous audited snapshots."""
import csv,json,hashlib,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'runs/rebuttal/closure_20261011';OLD=R/'docs/rebuttal/evidence/20261011_final';DEST=R/'docs/rebuttal/evidence/20261011_closed'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def main():
 v=read(B/'unified/verification.json');assert v['status']=='complete';e=read(B/'dino_e4/verification.json');assert e['status']=='complete'
 records=[]
 for r in read(OLD/'manifest.json')['files']:
  p=OLD/r['file'];assert sha(p)==r['sha256']
  rel=Path(r['file'])
  if str(rel) in ['diagnostics/all_completed_metrics.csv','diagnostics/verification.json','fixed_gt/verification.json']:rel=Path('historical')/rel
  q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 for p in (B/'unified').rglob('*'):
  if p.is_file():q=DEST/p.relative_to(B/'unified');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 save(DEST/'fixed_gt/verification.json',dict(v,scope='Corrected seed0 fixed-GT table; DINO geometry aligned; 17 methods, 136 rows, 34 endpoints'))
 # Full current proposal subsets, replacing cached legacy rows.
 for stem in ['curves','endpoints']:
  p=DEST/'fixed_gt'/(stem+'.csv');rr=list(csv.DictReader(p.open()));sub=[r for r in rr if r['method'] in ['sam_amg_clip','sam_amg_siglip','dinov2_sam_clip','dinov2_sam_siglip']]
  with (DEST/'fixed_gt'/('proposal_seed0_'+stem+'.csv')).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(sub)
 selected=[B/'freeda_label_corruption_audit.json',B/'freeda_e4_pairing.json',B/'freeda_full/protocol.json',B/'freeda_full/verification.json',B/'freeda_full/per_image.jsonl',B/'freeda_full/native_vs_archive_counts.npy',B/'freeda_full/confusion.npy',B/'dino_geometry/full/protocol.json',B/'dino_geometry/full/complete.json',B/'dino_geometry/full/summary.json',B/'dino_e4/verification.json']
 for p in (B/'dino_geometry/full').glob('*/result.json'):selected.extend([p,p.parent/'replay_verification.json'])
 selected+=list((B/'freeda_reproduction').glob('*'))
 selected += [p for p in (B/'code_snapshot').rglob('*') if p.is_file()]
 selected+=list((B/'dino_e4').glob('*/*/protocol.json'))+list((B/'dino_e4').glob('*/*/result.json'))+list((B/'coco_diagnostics').glob('*/result.json'))
 if (B/'freeda_provenance.json').exists():selected.append(B/'freeda_provenance.json')
 for p in selected:
  q=DEST/'closure'/p.relative_to(B);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 jobs=[j for j in read(R/'runs/rebuttal/execution_20261007/jobs.json') if j['method'].startswith('dinov2') and j['seed']==0];active=[]
 for j in jobs:
  dest=B/'dino_geometry/full'/j['name'];sources=[Path(j['out'])/n for n in ['manifest.jsonl','resolved_configuration.json','text_features.npy','input_hashes.json','plan.json']]
  sources+=[dest/'replay_verification.json',dest/'result.json']
  active.append(dict(method=j['method'],dataset=j['dataset'],candidate_seed=0,levels=j['sizes'],prediction_root=str(dest),original_job=str(j['out']),source_sha256={str(p):sha(p) for p in sources}))
 active.append(dict(method='freeda',dataset='coco_stuff171',prediction_root=str(B/'freeda_full/predictions'),verification=str(B/'freeda_full/verification.json'),prediction_encoding='native 0..170'))
 save(DEST/'closure/active_result_sources.json',dict(status='verified',updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),replacements=active,unchanged_sources='Retained rows use source_sha256 in current curves/common_gt tables; original snapshot remains immutable',invalid_for_current_quality=['FreeDA old archived COCO PNGs','DINO geometry-legacy predictions, including historical seed1/2'],efficiency='DINO current TF32-on pairing in closure/dino_e4; historical DINO timing must not be paired to corrected quality'))
 for p in sorted(DEST.rglob('*')):
  if p.is_file() and p.name!='manifest.json':records.append(dict(file=str(p.relative_to(DEST)),sha256=sha(p),bytes=p.stat().st_size))
 save(DEST/'manifest.json',dict(snapshot_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),previous_snapshot=str(OLD),previous_snapshot_sha256=sha(OLD/'manifest.json'),closure_verification=v,files=records))
 pointer=R/'runs/rebuttal/all_fixed_gt_20261009/current_main_report.json';old=read(pointer)
 if not (B/'previous_main_report.json').exists():save(B/'previous_main_report.json',old)
 old.update(main_endpoints=str(DEST/'fixed_gt/endpoints.csv'),main_curves=str(DEST/'fixed_gt/curves.csv'),coverage=str(DEST/'fixed_gt/coverage.csv'),verification=str(DEST/'verification.json'),active_sources=str(DEST/'closure/active_result_sources.json'),supplementary_three_seed='Historical geometry-legacy implementation only; not current corrected DINO evidence')
 save(pointer,old)
 print('SNAPSHOT COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
