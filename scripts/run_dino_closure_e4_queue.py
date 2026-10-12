"""Wait for quality repairs, then time only the two changed DINO configurations."""
import json,subprocess,time
from pathlib import Path
R=Path('/data/tianyi/TF-OVOS');B=R/'runs/rebuttal/closure_20261011'
while not all(p.exists() for p in [B/'dino_geometry/full/complete.json',B/'freeda_full/verification.json']):time.sleep(15)
results=[]
for method in ['dinov2_sam_clip','dinov2_sam_siglip']:
 for ds in ['voc20','context59','ade150','coco_stuff171']:
  subprocess.run(['bash',str(R/'scripts/python_e4'),str(R/'scripts/measure_dino_geometry_e4.py'),'--method',method,'--dataset',ds],check=True,cwd=R)
  results.append(json.loads((B/'dino_e4'/method/ds/'result.json').read_text()))
summary=[]
for method in ['dinov2_sam_clip','dinov2_sam_siglip']:
 rows=[r for x in results if x['method']==method for r in x['records']]
 summary.append(dict(method=method,images=len(rows),exact_matches=sum(r['changed_pixels']==0 for r in rows),mean_seconds=sum(r['seconds'] for r in rows)/len(rows),peak_allocated_mib=max(r['peak_allocated_mib'] for r in rows)))
(B/'dino_e4/verification.json').write_text(json.dumps(dict(status='complete',jobs=len(results),paired_images=sum(r['images'] for r in summary),exact_matches=sum(r['exact_matches'] for r in summary),methods=summary),indent=2)+'\n')
