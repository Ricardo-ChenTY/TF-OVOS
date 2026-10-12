"""Create a new immutable review package; never launch or overwrite experiments."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'docs/rebuttal/evidence/20261011_closed'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit('Refusing to overwrite an existing review package')
    # Validate frozen evidence before copying; exact bytes remain unchanged.
    frozen_manifest = json.loads((FROZEN/'manifest.json').read_text())
    for r in frozen_manifest['files']:
        if sha(FROZEN/r['file']) != r['sha256']:
            raise ValueError('Frozen evidence changed: '+r['file'])
    out.mkdir(parents=True)
    shutil.copytree(FROZEN, out/'evidence')
    for name in ['reviewer_experiment_summary_20261009.md','README.md','closure_review_20261011.md']:
        target=out/'report'/name;target.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'docs/rebuttal'/name,target)
    base = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    # Tracked source base plus executed source; exclude historical runs and paper assets.
    archive = subprocess.check_output(['git','archive',base,'src','scripts','tests','configs','pyproject.toml','requirements.txt','environment.yml','README.md','third_party_patches'], cwd=ROOT)
    code = out/'code';code.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(code, filter='data')
    shutil.copytree(FROZEN/'closure/code_snapshot/repository',code,dirs_exist_ok=True)
    # Include earlier prerequisite repairs that were still uncommitted when
    # experiments ran (diagnostic scripts are loaded dynamically by tests).
    modified = subprocess.check_output(['git','diff','--name-only',base], cwd=ROOT, text=True).splitlines()
    for rel in modified:
        if not rel.startswith(('src/','scripts/','tests/','configs/')):
            continue
        src=ROOT/rel; frozen=FROZEN/'closure/code_snapshot/repository'/rel
        if frozen.exists() and sha(src) != sha(frozen):
            raise ValueError('Current code differs from executed snapshot: '+rel)
        if src.is_file():
            target=code/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    entry = ['finalize_geometry_closure.py','snapshot_geometry_closure.py','render_geometry_closure_report.py',
             'validate_geometry_closure_report.py','run_dino_closure_e4_queue.py',
             'audit_freeda_saved_coco.py','probe_freeda_reproducibility.py',
             'build_closure_review.py','verify_closure_review.py']
    names = set(entry) | {p.name for p in (code/'scripts').glob('*.py')}
    pending = list(names); copied=[]
    while pending:
        name=pending.pop();src=ROOT/'scripts'/name
        if not src.exists():continue
        target=code/'scripts'/name
        # Keep frozen executed scripts as authoritative; add missing dependencies.
        if name in entry or not target.exists():shutil.copy2(src,target);copied.append(name)
        for node in ast.walk(ast.parse(target.read_text())):
            modules = [node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
            for module in modules:
                dep=(module or '').split('.')[0]+'.py'
                if (ROOT/'scripts'/dep).exists() and dep not in names:names.add(dep);pending.append(dep)
    shutil.copytree(ROOT/'configs/label_maps',code/'configs/label_maps',dirs_exist_ok=True)
    # Show the exact executed repository changes relative to the public base.
    tracked = subprocess.check_output(['git','ls-tree','-r','--name-only',base],cwd=ROOT,text=True).splitlines()
    import difflib
    diff=[]
    for path in sorted(p for p in code.rglob('*') if p.is_file() and p.suffix in {'.py','.json','.yaml','.yml'}):
        rel=path.relative_to(code).as_posix()
        if not rel.startswith(('src/','scripts/','tests/','configs/')):continue
        old=subprocess.check_output(['git','show',base+':'+rel],cwd=ROOT).decode() if rel in tracked else ''
        new=path.read_text()
        if old!=new:diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel if rel in tracked else '/dev/null',tofile='b/'+rel))
    write(out/'changes.patch',''.join(diff))
    # Relocate the already-used report checker; keep all numeric/table checks.
    checker=(ROOT/'scripts/validate_geometry_closure_report.py').read_text()
    checker=checker.replace('def main():','def main(root):\n global R,B,D\n R=root;B=root/"tools";D=root/"evidence"')
    checker=checker.replace("R/'docs/rebuttal/reviewer_experiment_summary_20261009.md'", "R/'report/reviewer_experiment_summary_20261009.md'")
    checker=checker.replace(" for p,h in sources.items():assert sha(p)==h,p", " # Original source hashes were checked locally before packaging; no raw data dependency here.")
    checker=checker.replace(" (B/'report_validation.json').write_text(json.dumps(result,indent=2)+'\\n');print(json.dumps(result,indent=2))", " return result")
    write(out/'tools/check_report.py',checker)
    shutil.copy2(ROOT/'scripts/render_geometry_closure_report.py',out/'tools/render_geometry_closure_report.py')
    shutil.copy2(ROOT/'runs/rebuttal/closure_20261011/report_before_closure.md',out/'tools/report_before_closure.md')
    shutil.copy2(ROOT/'scripts/verify_closure_review.py',out/'verify.py')
    shutil.copy2(ROOT/'runs/rebuttal/closure_20261011/report_validation.json',out/'local_report_validation.json')
    readme=(ROOT/'docs/rebuttal/closure_review_20261011.md').read_text()
    write(out/'README.md',readme)
    records=[dict(file=p.relative_to(out).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.rglob('*')) if p.is_file()]
    manifest=dict(schema=1,base_commit=base,frozen_evidence_manifest_sha256=sha(FROZEN/'manifest.json'),
                  policy='Frozen results; no new GPU jobs; full original PNGs remain local',files=records)
    write(out/'manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(out),'files':len(records),'bytes':sum(r['bytes'] for r in records),'base_commit':base},indent=2))

if __name__=='__main__':main()
