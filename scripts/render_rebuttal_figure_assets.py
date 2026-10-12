#!/usr/bin/env python3
"""Verify and render the portable figure bundle without models, data mounts or network."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def overlay(image, mask, color, alpha):
    mask=mask.resize(image.size,Image.Resampling.NEAREST).convert('L')
    colored=Image.composite(Image.new('RGB',image.size,color),image,mask)
    return Image.blend(image,colored,alpha)

def fit(canvas,image,box):
    image=image.copy();image.thumbnail((box[2]-box[0],box[3]-box[1]),Image.Resampling.LANCZOS)
    canvas.paste(image,(box[0]+(box[2]-box[0]-image.width)//2,box[1]+(box[3]-box[1]-image.height)//2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path);p.add_argument('--out',type=Path);p.add_argument('--verify-only',action='store_true');args=p.parse_args()
    root=args.bundle
    manifest=json.loads((root/'manifest.json').read_text())
    for name,info in manifest.items():
        path=root/name;assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size==info['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==info['sha256'],name
    cases=json.loads((root/'cases.json').read_text())['cases'];print('Verified',len(manifest),'files;',len(cases),'cases')
    if args.verify_only:return
    out=args.out or root/'rendered';out.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('DejaVuSans.ttf',15)
    except OSError:font=ImageFont.load_default()
    report=[]
    for c in cases:
        im=Image.open(root/c['input']).convert('RGB');target=Image.open(root/c['target_mask']).convert('L')
        labels=[s.strip() for s in (root/c['vocab']).read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        tiles=[('Input','',im),('Target',c['target'],overlay(im,target,(230,48,36),.45))]
        decoded=[]
        for method,spec in c['predictions'].items():
            a=np.asarray(Image.open(root/spec['path'])).astype('int32')+spec['add_to_raw_id']
            t=np.asarray(target.resize((a.shape[1],a.shape[0]),Image.Resampling.NEAREST))>0
            ids,counts=np.unique(a[t],return_counts=True);k=int(ids[counts.argmax()]);name=labels[k] if k>=0 else 'background/rejection'
            assert k==spec['selected_class_id'] and name==spec['selected_class_name']
            mask=Image.fromarray((a==k).astype('uint8')*255)
            tiles.append((method,f'{k}: {name}',overlay(im,mask,(34,190,54),.48)))
            decoded.append(dict(case=c['key'],method=method,id=k,name=name))
        canvas=Image.new('RGB',(280*len(tiles),620),'white');draw=ImageDraw.Draw(canvas)
        heading=c['dataset']+' / '+c['image_id']+' / '+c['target']
        draw.text((10,6),heading,font=font,fill='black')
        if c.get('warning'):draw.text((10,28),'HISTORICAL TARGET ONLY: absent in current GT; not ready for manuscript.',font=font,fill='red')
        t=np.asarray(target.resize(im.size,Image.Resampling.NEAREST));ys,xs=np.where(t>0)
        box=(max(0,int(xs.min())-25),max(0,int(ys.min())-25),min(im.width,int(xs.max())+26),min(im.height,int(ys.max())+26))
        for i,(method,caption,tile) in enumerate(tiles):
            x=280*i;draw.text((x+8,53),method,font=font,fill='black');fit(canvas,tile,(x+8,78,x+272,330))
            draw.text((x+8,338),caption.split(',')[0],font=font,fill='black');fit(canvas,tile.crop(box),(x+8,375,x+272,605))
        canvas.save(out/(c['key']+'.png'));report.extend(decoded)
    (out/'decoded_labels.json').write_text(json.dumps(report,indent=2)+'\n');print('Rendered',len(cases),'audit panels (not automatic replacements for manuscript figures)')
if __name__=='__main__':main()
