#!/usr/bin/env python3
"""Export both qualitative case sets, original maps, vocabularies and ADE polygons."""
import argparse
import ast
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
def norm(s): return re.sub('[^a-z0-9]+','',s.lower())
def polygon(obj,size):
    im=Image.new('L',size);ImageDraw.Draw(im).polygon(list(zip(obj['polygon']['x'],obj['polygon']['y'])),fill=255)
    return im

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-root',type=Path,required=True)
    parser.add_argument('--ade-archive',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    jobs={j['dataset']:j for j in json.loads((ROOT/'runs/rebuttal/execution_20261007/jobs.json').read_text()) if j['method']=='sam_amg_siglip' and j['seed']==0}
    aliases={'ade20k':'ade150','coco_stuff164k':'coco_stuff171'}
    mixed=json.loads((args.historical_root/'runs/analysis/qual_main_four_events_fixed_cases.json').read_text())
    cases=[dict(key='mixed_'+c['dataset']+'_'+Path(c['image_id']).name,set='mixed_dataset_cases',dataset=c['dataset'],image_id=Path(c['image_id']).name,target=c['target_name'],target_id=c['target_label']) for c in mixed]
    tree=ast.parse((ROOT/'TMLR26_Bench/scripts/make_independent_case_studies.py').read_text())
    old=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CASES' for t in n.targets))
    cases += [dict(key=k,set='submitted_all_ade_cases',dataset=c['dataset'],image_id=Path(c['filename']).stem,target=c['target'],raw_name=c['raw_name']) for k,c in old.items()]
    cached={d:rows(Path(j['out'])/'manifest.jsonl') for d,j in jobs.items()}
    report=[]
    with zipfile.ZipFile(args.ade_archive) as archive:
        annotations={Path(n).stem:n for n in archive.namelist() if '/validation/' in n and n.endswith('.json')}
        for c in cases:
            ds=aliases.get(c['dataset'],c['dataset']);job=jobs[ds]
            i,row=next((i,r) for i,r in enumerate(cached[ds]) if Path(r['image_id']).name==c['image_id'])
            dest=out/'cases'/c['key'];dest.mkdir(parents=True,exist_ok=True)
            def copy(src,name):
                src=Path(src);p=dest/name;shutil.copyfile(src,p);assert sha(src)==sha(p)
                return p.relative_to(out).as_posix()
            c['input']=copy(row['image_path'],'input.jpg');c['gt']=copy(row['mask_path'],'gt.png')
            vocab=Path(job['base_vocab']); labels=[x.strip() for x in vocab.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
            vd=out/'vocab'/vocab.name;vd.parent.mkdir(exist_ok=True);shutil.copyfile(vocab,vd);c['vocab']=vd.relative_to(out).as_posix()
            c['num_classes']=len(labels);c['gt_void']=65535 if len(labels)>256 else 255;c['gt_encoding']='zero_based'
            gt=np.asarray(Image.open(out/c['gt']))
            if 'raw_name' in c:
                member=annotations[c['image_id']];raw=archive.read(member);(dest/'ade_annotation.json').write_bytes(raw)
                ann=json.loads(raw)['annotation'];size=(ann['imsize'][1],ann['imsize'][0])
                objects=[o for o in ann['object'] if norm(o.get('raw_name',''))==norm(c['raw_name'])]
                if not objects:objects=[o for o in ann['object'] if norm(c['raw_name']) in norm(o.get('name',''))]
                obj=max(objects,key=lambda o:np.count_nonzero(np.asarray(polygon(o,size))))
                target=polygon(obj,size);c['target_protocol']='largest matching raw-name instance polygon from ADE source annotation'
                c['annotation']=(dest/'ade_annotation.json').relative_to(out).as_posix();c['annotation_archive_member']=member;c['target_object_id']=obj['id']
                c['target_id']=next(k for k,label in enumerate(labels) if norm(label.split(',')[0])==norm(c['target']))
            else:
                target=Image.fromarray((gt==c['target_id']).astype('uint8')*255);c['target_protocol']='whole-image semantic GT class mask'
                if not np.asarray(target).any():
                    legacy=ROOT/'data/raw/coco_stuff171_labels'/(c['image_id']+'.png')
                    assert ds=='coco_stuff171' and legacy.exists()
                    c['historical_gt']=copy(legacy,'historical_gt.png')
                    oldgt=np.asarray(Image.open(legacy));target=Image.fromarray((oldgt==c['target_id']).astype('uint8')*255)
                    c['target_protocol']='historical GT only; target is absent in current GT'
                    c['warning']='Do not present this target as current GT flower: historical target ID 107 is absent from current corrected GT. Select a new COCO case or revise target after checking source annotation.'
                    ids,counts=np.unique(gt[oldgt==c['target_id']],return_counts=True)
                    c['current_gt_on_historical_target']=[dict(id=int(k),name=labels[int(k)] if k<len(labels) else 'void',pixels=int(n)) for k,n in zip(ids,counts)]
            assert np.asarray(target).any();target.save(dest/'target.png');c['target_mask']=(dest/'target.png').relative_to(out).as_posix()
            c['predictions']={}
            for method in ['corrclip','naclip','proxyclip','sclip','sam_amg_siglip']:
                if method=='sam_amg_siglip':
                    src=Path(job['out'])/f"candidates_{job['sizes'][0]}"/f'{i:06d}.png';offset=0
                else:
                    src=args.historical_root/'runs/artifacts/official_predictions'/method/c['dataset']/(c['image_id']+'.png')
                    offset=-1 if ds in ['voc20','context59','ade150'] else 0
                p=copy(src,method+'.png');a=np.asarray(Image.open(out/p)).astype('int32')+offset
                assert a.min()>=-1 and a.max()<len(labels),(c['key'],method,a.min(),a.max())
                mask=np.asarray(target.resize((a.shape[1],a.shape[0]),Image.Resampling.NEAREST))>0
                ids,counts=np.unique(a[mask],return_counts=True);majority=int(ids[counts.argmax()])
                name=labels[majority] if majority>=0 else 'background/rejection'
                c['predictions'][method]=dict(path=p,add_to_raw_id=offset,rejection_id=-1,selected_class_id=majority,selected_class_name=name,width=a.shape[1],height=a.shape[0],source='corrected_seed0_evaluation' if method=='sam_amg_siglip' else 'historical_dense_export')
            report.append(c);print(c['key'],'exported',flush=True)
    metadata=dict(version=1,cases=report,old_hf_cache_recovered=False,
                  annotation_replacement='ADE source ZIP annotations; not claimed byte-identical to the missing HF cache',
                  scope='Selected figure assets only, not a full prediction or dataset backup. Final appendix PDF correspondence remains unverified.')
    (out/'cases.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest={p.relative_to(out).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(len(report),'cases',len(manifest),'files',sum(r['bytes'] for r in manifest.values()),'bytes')
if __name__=='__main__':main()
