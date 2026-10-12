"""Check every replaced report table against corrected machine-readable evidence."""
import csv,json,re,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'runs/rebuttal/closure_20261011';D=R/'docs/rebuttal/evidence/20261011_closed'
def rows(p):return list(csv.DictReader(Path(p).open()))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b):assert abs(float(a)-float(b))<=.000051,(a,b)
def table(s):
 t=re.search(r'(?m)^\| .*\n\| [- |]+\n((?:\| .*\n)+)',s);assert t
 return [[x.strip() for x in line.strip().strip('|').split('|')] for line in t.group(1).splitlines()]
def section(s,label):
 pos=s.index(label);end=s.find('\n### ',pos+len(label));return s[pos:end if end!=-1 else len(s)]
def main():
 from render_geometry_closure_report import NAMES
 reverse={v:k for k,v in NAMES.items()};s=(R/'docs/rebuttal/reviewer_experiment_summary_20261009.md').read_text();old=(B/'report_before_closure.md').read_text()
 assert set(re.findall(r'<a id="([^"]+)',s))==set(re.findall(r'<a id="([^"]+)',old))
 assert not any(x in s for x in ['来源差异尚未完全归因','保留原 baseline 的特征裁剪/网格对应行为','技术收尾正在进行'])
 ix={(r['method'],r['dataset']):r for r in rows(D/'fixed_gt/endpoints.csv')};t=table(section(s,'### O2：'));assert len(t)==17
 for row in t:
  for j,ds in enumerate(['ade150','context59']):
   r=ix[reverse[row[0]],ds]
   for i,k in enumerate(['base_miou_percent','expanded_miou_percent','drop_pp']):near(row[1+j*3+i],r[k])
 o3=section(s,'### O3：');tables=re.findall(r'(?m)^\| 方法 \|.*\n\| [- |]+\n(?:\| .*\n)+',o3);assert len(tables)==2
 ix={(r['method'],r['dataset']):r for r in rows(D/'diagnostics/common_gt_counts.csv') if float(r['iou_threshold'])==.5}
 for ds,raw in zip(['context459','ade847'],tables):
  t=table(raw);assert len(t)==14
  for rr in t:
   r=ix[reverse[rr[0]],ds]
   for i,k in enumerate(['gt_pairs','localized_correct','localized_wrong','unlocalized']):assert int(rr[i+1])==int(r[k])
   for i,k in [(5,'coverage'),(6,'mcmr')]:assert abs(float(rr[i])-100*float(r[k]))<=.00501
 ix={(r['method'],r['group']):r for r in rows(D/'diagnostics/coco_thing_stuff_intersection.csv')};t=table(section(s,'### O4：'));assert len(t)==8
 for rr in t:
  a=float(ix[reverse[rr[0]],'thing']['miou_percent']);b=float(ix[reverse[rr[0]],'stuff']['miou_percent']);near(rr[1],a);near(rr[2],b);near(rr[3],a-b)
 ix={(r['method'],r['dataset']):r for r in rows(D/'diagnostics/submitted_comparison.csv')};t=table(section(s,'### 原生数据集重跑与固定 GT 控制分别引用'));assert len(t)==24
 for rr in t:
  r=ix[reverse[rr[1]],rr[0]]
  for i,k in [(2,'submitted_miou'),(3,'new_miou'),(4,'change_pp')]:near(rr[i],r[k])

 ix={r['method']:r for r in rows(D/'diagnostics/compact_mean_worst.csv')};t=table(section(s,'### O5：'));assert len(t)==4
 for rr in t:
  r=ix[reverse[rr[0]]]
  for i,k in enumerate(['voc20','context59','ade150','coco_stuff171','mean_miou_percent'],1):near(rr[i],r[k])
  worst,ds=rr[6].split(' / ');near(worst,r['worst_miou_percent']);assert ds==r['worst_dataset']
 manifest=json.loads((D/'manifest.json').read_text())
 for r in manifest['files']:assert sha(D/r['file'])==r['sha256'],r['file']
 sources={}
 for name in ['fixed_gt/curves.csv','diagnostics/common_gt_counts.csv']:
  for r in rows(D/name):sources[r['source']]=r['source_sha256']
 for p,h in sources.items():assert sha(p)==h,p
 # All named local evidence paths must exist, including regenerated plots.
 for name in re.findall(r'`((?:fixed_gt|diagnostics|e4_sample|e4_targeted|component_min128|component_min64|ade847_corrected|closure)/[^`]+)`',s):assert (D/name).exists(),name
 result=dict(status='passed',reviewer_anchors=len(set(re.findall(r'<a id="([^"]+)',s))),verified_fixed_gt_rows=17,verified_common_gt_rows=28,verified_coco_rows=8,verified_native_comparison_rows=24,snapshot_hashes=len(manifest['files']),source_hashes=len(sources),report_sha256=sha(R/'docs/rebuttal/reviewer_experiment_summary_20261009.md'))
 (B/'report_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
