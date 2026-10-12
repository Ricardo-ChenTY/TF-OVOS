#!/usr/bin/env python3
"""Rescore existing MaskCLIP maps through the released common-GT evaluator (CPU)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import generate_diagnostic_tables as diagnostic

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('maskclip', 'maskclip_attn', 'maskclip_attn_slide')
DATASETS = {'context459': 'context459_val', 'ade847': 'ade20k847_val'}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def run(task):
    source, out, method, dataset = task
    folder = source / method / DATASETS[dataset]
    rows = [json.loads(line) for line in (folder / 'predictions.jsonl').read_text().splitlines()]
    mapping = {r['image_id']: folder / r['mask_path'] for r in rows}
    assert len(mapping) == len(rows)
    spec = diagnostic._dataset_spec(dataset)
    samples = diagnostic.load_manifest(spec.manifest)
    assert set(mapping) == {s.image_id for s in samples}
    records = []
    for sample in samples:
        path = mapping[sample.image_id]
        with Image.open(path) as im:
            a = np.asarray(im)
            assert a.ndim == 2 and a.min() >= 0 and a.max() < spec.num_classes
            assert im.mode in ('I;16', 'I')
            records.append(dict(image_id=sample.image_id, prediction=path.relative_to(source).as_posix(),
                                sha256=sha(path), width=im.width, height=im.height,
                                min_id=int(a.min()), max_id=int(a.max()),
                                gt_sha256=sha(sample.mask_path)))
    diagnostic._prediction_path = lambda unused, image_id: mapping[image_id]
    summary, mismatches = diagnostic._analyse_method_dataset(
        method, folder, spec, diagnostic.read_vocab(spec.vocab), False, None, 0)
    assert summary['missing_images'] == 0
    assert summary['gt_class_regions'] == {'context459': 33026, 'ade847': 20369}[dataset]
    result = dict(summary=summary, input_files=records,
                  prediction_manifest_sha256=sha(folder / 'predictions.jsonl'),
                  vocabulary_sha256=sha(spec.vocab),
                  prediction_encoding='zero_based_uint16', new_inference=False,
                  historical_adapter=rows[0].get('metadata'),
                  geometry='full-image output, nearest-neighbor resize to current GT; no center crop',
                  evaluator_sha256=sha(ROOT/'scripts/generate_diagnostic_tables.py'),
                  metric_code_sha256=sha(ROOT/'src/tf_ovos/metrics.py'))
    (out / f'{method}_{dataset}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(method, dataset, summary['processed_images'], summary['miou'], flush=True)
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    tasks = [(args.source,args.out,m,d) for m in METHODS for d in DATASETS]
    with ProcessPoolExecutor(max_workers=3) as pool:
        summaries = list(pool.map(run, tasks))
    with (args.out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    counts=[]
    for r in summaries:
        for suffix, threshold in [('05',0.5),('075',0.75)]:
            n=r['gt_class_regions']; loc=r[f'localized_pairs_at_{suffix}']; wrong=r[f'mismatch_pairs_at_{suffix}']
            counts.append(dict(method=r['method'],dataset=r['dataset'],iou_threshold=threshold,
                               gt_pairs=n,localized_correct=loc-wrong,localized_wrong=wrong,unlocalized=n-loc,
                               coverage=loc/n,mcmr=wrong/loc if loc else None))
    with (args.out/'common_gt_counts.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(counts[0]));w.writeheader();w.writerows(counts)
if __name__=='__main__':main()
