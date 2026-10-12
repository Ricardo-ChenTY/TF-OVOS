"""Verify a portable closure review bundle without GPU or original data paths."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = {r['file'] for r in manifest['files']}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and p != root / 'manifest.json'
              and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    require(actual == expected, 'Bundle file inventory differs')
    for r in manifest['files']:
        p = root / r['file']
        require(p.resolve().is_relative_to(root.resolve()), 'Unsafe manifest path')
        require(p.stat().st_size == r['bytes'] and sha(p) == r['sha256'], r['file'])
    d = root / 'evidence'
    def read(name):
        return json.loads((d / name).read_text())
    def rows(name):
        with (d / name).open() as stream:
            return list(csv.DictReader(stream))
    def near(a, b, tolerance=1e-8):
        require(abs(float(a) - float(b)) <= tolerance, f'{a} != {b}')
    # Check the original frozen manifest too, including nested manifests omitted
    # by its historical generator in the new outer inventory above.
    for r in read('manifest.json')['files']:
        require(sha(d / r['file']) == r['sha256'], r['file'])
    curves = rows('fixed_gt/curves.csv')
    ends = rows('fixed_gt/endpoints.csv')
    require(len(curves) == 136 and len(ends) == 34, 'Fixed GT coverage')
    require(len({r['method'] for r in curves}) == 17, 'Method coverage')
    index = {(r['method'], r['dataset'], int(r['candidates'])): r for r in curves}
    require(len(index) == 136, 'Duplicate curves')
    increases = []
    for r in ends:
        levels = [150, 300, 500, 847] if r['dataset'] == 'ade150' else [59, 150, 300, 459]
        values = [float(index[r['method'], r['dataset'], k]['miou_percent']) for k in levels]
        near(values[0], r['base_miou_percent']); near(values[-1], r['expanded_miou_percent'])
        near(values[0] - values[-1], r['drop_pp'])
        require(values[0] > values[-1], 'Non-descending endpoint')
        for a, b, x, y in zip(levels, levels[1:], values, values[1:]):
            if y > x:
                increases.append((r['method'], r['dataset'], a, b, y - x))
    steps = rows('fixed_gt/nonmonotonic_steps.csv')
    require(len(increases) == len(steps) == 4, 'Nonmonotonic steps')
    for method, ds, a, b, delta in increases:
        match = [r for r in steps if (r['method'], r['dataset'], int(r['from_candidates']), int(r['to_candidates'])) == (method, ds, a, b)]
        require(len(match) == 1, 'Missing nonmonotonic step'); near(delta, match[0]['increase_pp'])
    common = rows('diagnostics/common_gt_counts.csv')
    require(len(common) == 56, 'Common GT coverage')
    for r in common:
        n, c, w, u = [int(r[k]) for k in ('gt_pairs', 'localized_correct', 'localized_wrong', 'unlocalized')]
        require(n == c + w + u, 'Common denominator')
        near((c+w)/n, r['coverage']); near(w/(c+w), r['mcmr'])
    coco = rows('diagnostics/coco_thing_stuff_intersection.csv')
    require(len(coco) == 16, 'COCO coverage')
    for method in {r['method'] for r in coco}:
        t, s = [next(r for r in coco if r['method'] == method and r['group'] == group) for group in ('thing', 'stuff')]
        require(int(t['images']) == int(s['images']) == 4894, 'COCO image set size')
        require(float(t['miou_percent']) > float(s['miou_percent']), 'Thing/stuff result')
    freeda = read('closure/freeda_full/verification.json')
    require(freeda['status'] == 'complete' and freeda['images'] == 5000 and freeda['changed_non255_pixels'] == 0, 'FreeDA completion')
    near(freeda['miou_percent'], 28.796316822156953)
    complete = read('closure/dino_geometry/full/complete.json')
    require(complete['jobs'] == 12 and complete['metric_rows'] == 24, 'DINO completion')
    replay = list((d / 'closure/dino_geometry/full').glob('*/replay_verification.json'))
    require(len(replay) == 12, 'DINO replay coverage')
    for p in replay:
        r = json.loads(p.read_text())
        require(r['all_old_pngs_exact'] and r['images'] == len(r['records']) and r['max_legacy_dino_score_error'] <= 2e-5, str(p))
    e4 = read('closure/dino_e4/verification.json')
    require(e4['jobs'] == 8 and e4['paired_images'] == e4['exact_matches'] == 200, 'E4 pairing')
    for r in e4['methods']:
        jobs = [json.loads(p.read_text()) for p in (d/'closure/dino_e4'/r['method']).glob('*/result.json')]
        require(len(jobs) == 4, 'E4 per-dataset coverage')
        require(sum(j['images'] for j in jobs) == sum(j['exact_matches'] for j in jobs) == r['images'] == r['exact_matches'] == 100, 'E4 per-image totals')
        near(sum(j['mean_seconds']*j['images'] for j in jobs)/100, r['mean_seconds'])
        near(max(j['peak_allocated_mib'] for j in jobs), r['peak_allocated_mib'])
    # Run the original table checker with relocated evidence and report only.
    # Its source-file checks are separately covered by local validation; those
    # original predictions/GT are intentionally not required by this package.
    import importlib.util
    import sys
    tools = root / 'tools'
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location('portable_report_check', tools / 'check_report.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.main(root)
    return {'status': 'passed', 'bundle_files': len(expected), 'fixed_gt_rows': 136,
            'descending_endpoints': 34, 'common_gt_rows': 56, 'coco_methods': 8,
            'dino_full_jobs': 12, 'efficiency_exact_pairs': 200,
            'scope': 'portable hashes, aggregate consistency and report tables; no raw PNG re-inference'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle.resolve()), indent=2))
