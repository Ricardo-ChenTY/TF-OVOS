#!/usr/bin/env python3
"""Read-only comparison of release presets, project configs and quality-run records."""
import ast
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
LEGACY=Path('/data/tianyi/TF-OVCOS')
REPOS=LEGACY/'third_party/official_methods'
OUT=ROOT/'docs/rebuttal/evidence/release_presets_20261013'
METHODS={'sclip':('SCLIP','clip_segmentor.py','CLIPForSegmentation'),
         'naclip':('NACLIP','naclip.py','NACLIP'),
         'resclip':('ResCLIP','resclip_segmentor.py','ResCLIPSegmentation'),
         'trident':('Trident','trident.py','Trident'),
         'cass':('CASS','cass_segmentor.py','CASS_segmentor')}
DATASETS=['voc20','context59','ade20k','coco_stuff164k','context459','ade847']

def git(repo,*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.DEVNULL)
def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def sha(p):return sha_bytes(Path(p).read_bytes())
def value(node,env):
    if isinstance(node,ast.Constant):return node.value
    if isinstance(node,ast.Name):return env[node.id]
    if isinstance(node,(ast.List,ast.Tuple)):return [value(n,env) for n in node.elts]
    if isinstance(node,ast.Dict):return {value(k,env):value(v,env) for k,v in zip(node.keys,node.values)}
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='dict':return {k.arg:value(k.value,env) for k in node.keywords}
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):return -value(node.operand,env)
    raise ValueError(ast.dump(node))
def parse(text):
    env={}
    for n in ast.parse(text).body:
        if isinstance(n,ast.Assign):
            try:v=value(n.value,env)
            except (ValueError,KeyError):continue
            for t in n.targets:
                if isinstance(t,ast.Name):env[t.id]=v
    return env

def merge(a,b):
    result=dict(a)
    for k,v in b.items():
        result[k]=merge(result.get(k,{}),v) if isinstance(v,dict) and isinstance(result.get(k,{}),dict) else v
    return result

def cfg(repo,path,upstream):
    text=git(repo,'show','HEAD:'+path) if upstream else (repo/path).read_text()
    d=parse(text);bases=d.pop('_base_',[]);bases=[bases] if isinstance(bases,str) else bases;result={}
    for b in bases:result=merge(result,cfg(repo,str(Path(path).parent/b),upstream))
    return merge(result,d)

def defaults(text,cls):
    c=next(n for n in ast.parse(text).body if isinstance(n,ast.ClassDef) and n.name==cls)
    f=next(n for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');result={}
    for arg,n in zip(f.args.args[-len(f.args.defaults):],f.args.defaults):
        try:result[arg.arg]=value(n,{})
        except (ValueError,KeyError):pass
    return result

def latest_quality(method,ds):
    hits=[]
    for p in (LEGACY/'runs/official'/method).glob(ds+'*/**/*.log'):
        lines=[x for x in p.read_text(errors='replace').splitlines() if 'Iter(test)' in x and 'mIoU:' in x]
        config=p.parent/'vis_data/config.py'
        if lines and config.exists():hits.append((lines[-1][:19],p,config))
    return max(hits) if hits else None

def main():
    OUT.mkdir(parents=True,exist_ok=True);records=[];sources={}
    for method,(dirname,code,cls) in METHODS.items():
        repo=REPOS/dirname;commit=git(repo,'rev-parse','HEAD').strip();url=git(repo,'remote','get-url','origin').strip().removesuffix('.git')
        default=defaults(git(repo,'show','HEAD:'+code),cls)
        current_default=defaults((repo/code).read_text(),cls)
        sources[method]=dict(commit=commit,upstream=url,constructor_defaults_match_release=default==current_default,
                             native_code_matches_release=(repo/code).read_text()==git(repo,'show','HEAD:'+code))
        for ds in DATASETS:
            local=repo/'configs'/f'cfg_tfovos_{ds}.py'
            if not local.exists():local=repo/'configs'/f'cfg_tfovos_{ds}_e2.py'
            if not local.exists():continue
            upstream='configs/cfg_'+ds+'.py';origin='direct_dataset_release'
            try:official=cfg(repo,upstream,True)
            except subprocess.CalledProcessError:
                fallback={'context459':'context59','ade847':'ade20k'}.get(ds)
                if not fallback:raise
                upstream='configs/cfg_'+fallback+'.py';official=cfg(repo,upstream,True);origin='project_large_vocabulary_adaptation_from_compact_preset'
            project=cfg(repo,local.relative_to(repo).as_posix(),False)
            quality=latest_quality(method,ds);runtime=parse(quality[2].read_text()) if quality else {}
            native=merge(default,official['model'])
            # The released CASS eval.py defaults --pamr to off and overrides constructor 10.
            if method=='cass':native['pamr_steps']=0
            configured=merge(current_default,project['model']);actual=merge(current_default,runtime.get('model',project['model']))
            allowed={'name_path','sam_ckpt','dataset','device'}
            diffs={k:dict(release=native.get(k),project_config=configured.get(k),quality_record=actual.get(k)) for k in set(native)|set(configured)|set(actual) if k not in allowed and native.get(k)!=actual.get(k)}
            rec=dict(method=method,dataset=ds,upstream_commit=commit,upstream_config=upstream,upstream_url=url+'/blob/'+commit+'/'+upstream,
                     source_relation=origin,project_config=local.relative_to(LEGACY).as_posix(),project_sha256=sha(local),
                     release_model=native,project_model=configured,quality_model=actual,
                     release_pipeline=official.get('test_pipeline'),project_pipeline=project.get('test_pipeline'),quality_pipeline=runtime.get('test_pipeline'),
                     model_differences=diffs,release_cli_defaults_applied={'pamr_steps':0} if method=='cass' else {},quality_config=quality[2].relative_to(LEGACY).as_posix() if quality else None,
                     quality_config_sha256=sha(quality[2]) if quality else None,quality_log=quality[1].relative_to(LEGACY).as_posix() if quality else None,
                     quality_log_sha256=sha(quality[1]) if quality else None)
            # Preserve parameter semantics while removing irrelevant author/server path identities.
            for key in ['release_model','project_model','quality_model']:
                for field in ['sam_ckpt','name_path']:
                    if field in rec[key]:rec[key][field]=Path(rec[key][field]).name
            records.append(rec)
    result=dict(scope='Five dense methods, six targets; defaults resolved from release constructor. Latest completed historical quality log per dataset family, not proof of every published cell linkage.',sources=sources,records=records)
    (OUT/'dense_config_audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    table=[]
    for r in records:
        model=r['quality_model'];resize=next((x.get('scale') for x in r['quality_pipeline'] or [] if x['type']=='Resize'),None)
        table.append(dict(method=r['method'],dataset=r['dataset'],resize=json.dumps(resize),slide_crop=model.get('slide_crop'),slide_stride=model.get('slide_stride'),temp_thd=model.get('temp_thd'),h_threshold=model.get('h_threshold'),global_semantics_weight=model.get('global_semantics_weight'),mean_vector_weight=model.get('mean_vector_weight'),sam_model_type=model.get('sam_model_type'),sam_refinement=model.get('sam_refinement'),source_relation=r['source_relation'],model_differences=json.dumps(r['model_differences'],sort_keys=True),quality_config=r['quality_config']))
    with (OUT/'quality_settings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    for r in records:
        if r['model_differences']:print(r['method'],r['dataset'],r['model_differences'])
    import yaml
    repo=REPOS/'freeda';commit=git(repo,'rev-parse','HEAD').strip();freeda=[]
    for ds,key in [('voc20','pascal20'),('context59','pascal59'),('ade20k150','ade'),('coco_stuff171','cocostuff')]:
        path=LEGACY/'runs/freeda'/ds/'config.json'
        actual=yaml.safe_load(path.read_text());source=f'src/configs/{key}/freeda_{key}.yml'
        released=yaml.safe_load(git(repo,'show','HEAD:'+source))
        diff={k:dict(release=v,quality=actual['model'].get(k)) for k,v in released['model'].items() if v!=actual['model'].get(k)}
        freeda.append(dict(dataset=ds,commit=commit,upstream_config=source,
                           upstream_url='https://github.com/aimagelab/freeda/blob/'+commit+'/'+source,
                           quality_config=path.relative_to(LEGACY).as_posix(),quality_config_sha256=sha(path),
                           model_fields_compared=len(released['model']),model_differences=diff,
                           superpixel=actual['model']['superpixel'],
                           scope='Compares every field in released model section; not a proof of historical preset-selection procedure or all dataset/evaluator changes.'))
    (OUT/'freeda_config_audit.json').write_text(json.dumps(freeda,indent=2)+'\n')
    repo=REPOS/'Trident';code=git(repo,'show','HEAD:trident.py');base=cfg(repo,'configs/base_config.py',True)
    checkpoint=Path('/data/sdd/tianyi/TF-OVOS-rebuttal/.cache/e4/weights/sam_vit_b_01ec64.pth')
    pinned=next(x for x in json.loads((ROOT/'runs/rebuttal/e4_20261009/new_checkpoint_and_index_sha256.json').read_text()) if Path(x['path']).name==checkpoint.name)
    digest=sha(checkpoint);assert digest==pinned['sha256']
    trident=dict(upstream_commit=sources['trident']['commit'],official_sam_model_type=base['model']['sam_model_type'],
                 official_sam_filename=Path(base['model']['sam_ckpt']).name,official_sam_refinement=base['model']['sam_refinement'],
                 quality_runs=[dict(dataset=r['dataset'],sam_model_type=r['quality_model']['sam_model_type'],sam_checkpoint=r['quality_model']['sam_ckpt'],sam_refinement=r['quality_model']['sam_refinement'],quality_config=r['quality_config'],sha256=r['quality_config_sha256']) for r in records if r['method']=='trident'],
                 cached_checkpoint=dict(filename=checkpoint.name,bytes=checkpoint.stat().st_size,sha256=digest,matches_20261009_pinned_hash=True),
                 implementation_sha256=sha_bytes(code.encode()),native_implementation_matches_upstream=sources['trident']['native_code_matches_release'],
                 sam_usage='SAM encoder features, last attention and value tensors are used even with optional final SAM refinement disabled.',
                 declared_backbones='CLIP ViT-B/16 (OpenAI), DINO ViT-B/16, SAM ViT-B; final SAM refinement disabled',
                 limitations=['Historical May checkpoint path no longer exists; log proves model type and filename, not byte identity of the lost historical file.', 'Official README example enables --sam_refine; the evaluated no-refinement configuration is the released code default, not an assertion of identity to every README/paper result.'])
    assert trident['official_sam_model_type']=='vit_b' and all(r['sam_model_type']=='vit_b' and r['sam_refinement'] is False for r in trident['quality_runs'])
    (OUT/'trident_sam_audit.json').write_text(json.dumps(trident,indent=2)+'\n')
    print('Saved',len(records),'dense config comparisons, four FreeDA model comparisons and Trident SAM provenance')
if __name__=='__main__':main()
