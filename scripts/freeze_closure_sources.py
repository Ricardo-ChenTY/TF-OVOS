"""Freeze the concrete local repair source, native configuration and runtime versions."""
import json,hashlib,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R/'runs/rebuttal/closure_20261011';OUT=B/'code_snapshot';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=['scripts/repair_dino_geometry_from_cache.py','scripts/restore_freeda_coco_native.py','scripts/e4_special_backends.py','scripts/measure_dino_geometry_e4.py','scripts/measure_e4.py','scripts/python_e4','scripts/e4_full_common.py','scripts/e4_gpu_guard.py','scripts/pin_freeda_reproduction.py','tests/test_dino_geometry.py','tests/test_rebuttal_metrics.py','scripts/freeze_closure_sources.py']
files=[R/x for x in local]+list((R/'src/tf_ovos').rglob('*.py'));records=[]
for p in files:
 dest=OUT/'repository'/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);records.append(dict(source=str(p),snapshot=str(dest.relative_to(OUT)),sha256=sha(p)))
repo=Path('/data/tianyi/TF-OVCOS/third_party/official_methods/freeda/src')
for folder in ['models','segmentation','datasets','utils','us']:
 for p in (repo/folder).rglob('*.py'):
  dest=OUT/'native_freeda'/p.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);records.append(dict(source=str(p),snapshot=str(dest.relative_to(OUT)),sha256=sha(p)))
for folder in ['freeda','freeda_artifact']:
 p=Path('/data/tianyi/TF-OVCOS/runs')/folder/'coco_stuff171/config.json';dest=OUT/(folder+'_coco_config.yaml');shutil.copy2(p,dest);records.append(dict(source=str(p),snapshot=str(dest.relative_to(OUT)),sha256=sha(p)))
packages=['torch','torchvision','timm','open-clip-torch','transformers','mmcv','mmcv-full','mmsegmentation','numpy','Pillow','scikit-image','faiss-cpu','omegaconf']
code='import importlib.metadata as m,json,sys; d={"python":sys.version};\nfor p in '+repr(packages)+':\n try:d[p]=m.version(p)\n except m.PackageNotFoundError:pass\nprint(json.dumps(d))'
import os
versions={}
for name,python in [('dino','/data/tianyi/conda_envs/tf-ovos/bin/python'),('freeda','/data/tianyi/conda_envs/freeda-official-py38-t112b/bin/python')]:
 env=os.environ.copy();env['E4_PYTHON']=python;result=subprocess.run(['bash',str(R/'scripts/python_e4'),'-c',code],env=env,text=True,check=True,capture_output=True);versions[name]=json.loads(result.stdout)
(OUT/'manifest.json').write_text(json.dumps(dict(files=records,runtime_versions=versions,scope='Actual local repair code and native FreeDA model/configuration sources; external weights/index/prototypes pinned separately.'),indent=2)+'\n')
print('FROZEN',len(records),'files',json.dumps(versions))
