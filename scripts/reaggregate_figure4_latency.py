"""Reaggregate existing Figure 4 timings; stdlib only, no inference."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'docs/rebuttal/evidence/reply_support_20261012'
OUT = ROOT / 'docs/rebuttal/evidence/figure4_equal_target_20261013'
TARGETS = ('voc20', 'context59', 'ade150', 'coco_stuff171')

def read_csv(path):
    return list(csv.DictReader(path.read_text().splitlines()))

def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}
    def csv_source(rel):
        path = BASE / rel
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read_csv(path)
    old = csv_source('derived/current_quality_latency_pairs.csv')
    sam = csv_source('e4_targeted/proposal_same_recipe/latency_by_dataset.csv')
    corr = csv_source('e4_targeted/corrclip_online/latency_by_dataset.csv')
    archive = ROOT / 'docs/rebuttal/artifacts/review_20261011_closed.tar.gz'
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert archive_hash == '352f9200df173cb5d08d7ef1ee9423f22e0c3294cc1ef3c2ef766bd102d941a3'
    inputs[str(archive.relative_to(ROOT))] = archive_hash
    detail = []
    with tarfile.open(archive) as tar:
        for row in old:
            method = row['method']
            for dataset in TARGETS:
                if method.startswith('dinov2_'):
                    member = f'review_20261011_closed/evidence/closure/dino_e4/{method}/{dataset}/result.json'
                    raw = tar.extractfile(member).read()
                    r = json.loads(raw)
                    n, seconds = r['images'], r['mean_seconds']
                    assert n == r['exact_matches'] == len(r['records']) == 25
                    assert all(x['changed_pixels'] == 0 for x in r['records'])
                    assert math.isclose(statistics.mean(x['seconds'] for x in r['records']), seconds, abs_tol=1e-12)
                    source = f'{archive.name}:{member}'
                    inputs[source] = hashlib.sha256(raw).hexdigest()
                else:
                    candidates = corr if method.startswith('corrclip_') else [x for x in sam if x['method'] == method]
                    found = [x for x in candidates if x['dataset'] == dataset]
                    assert len(found) == 1
                    r = found[0]
                    n, seconds = int(r['images']), float(r['online_mean_seconds'])
                    source = 'e4_targeted/corrclip_online/latency_by_dataset.csv' if method.startswith('corrclip_') else 'e4_targeted/proposal_same_recipe/latency_by_dataset.csv'
                assert n > 0 and seconds > 0 and math.isfinite(seconds)
                detail.append(dict(method=method, dataset=dataset, timed_images=n,
                                   online_mean_seconds=seconds, target_weight=0.25, source=source))
    summary = []
    for row in old:
        per_target = [x for x in detail if x['method'] == row['method']]
        macro = statistics.mean(x['online_mean_seconds'] for x in per_target)
        pooled = sum(x['timed_images'] * x['online_mean_seconds'] for x in per_target) / sum(x['timed_images'] for x in per_target)
        assert math.isclose(pooled, float(row['mean_online_seconds']), abs_tol=1e-12)
        assert sum(x['timed_images'] for x in per_target) == int(row['timed_images'])
        summary.append(dict(method=row['method'], full_four_target_mean_miou_percent=float(row['full_four_target_mean_miou_percent']),
                            equal_target_online_seconds=macro, previous_online_seconds=float(row['mean_online_seconds']),
                            timed_images=int(row['timed_images']), target_weights='0.25 each',
                            sampling='full target splits' if row['method'].startswith('corrclip_') else '25 images per target',
                            scope='online inference under declared configuration; excludes offline preparation'))
    write_csv(OUT / 'latency_by_dataset.csv', detail)
    write_csv(OUT / 'quality_latency_pairs.csv', summary)
    (OUT / 'verification.json').write_text(json.dumps(dict(
        source_sha256=inputs, dataset_rows=len(detail), methods=len(summary),
        formula='equal_target_online_seconds = sum(mean online seconds for each of four targets) / 4',
        checks=['Previous aggregates reproduced from dataset means and image counts',
                'Corrected DINO means recomputed from all 200 archived per-image records',
                'Quality values copied unchanged from configuration-matched full-run summary'],
        limitations=['Equal target weights do not equalize sample sizes or sampled images',
                     'Not a universal end-to-end Pareto ranking; excludes offline preparation',
                     'No new inference; historical frozen evidence preserved'],
        output_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (OUT / 'latency_by_dataset.csv', OUT / 'quality_latency_pairs.csv')}
    ), indent=2) + '\n')
    for row in summary:
        print(row['method'], row['equal_target_online_seconds'])

if __name__ == '__main__':
    main()
