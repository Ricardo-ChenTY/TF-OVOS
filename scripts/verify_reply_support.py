"""Verify the browsable reply evidence and released runtime source, without GPU."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/rebuttal/evidence/reply_support_20261012'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    m = json.loads((DATA / 'manifest.json').read_text())
    for r in m['files']:
        p = DATA / r['file']
        if sha(p) != r['sha256'] or p.stat().st_size != r['bytes']:
            raise ValueError('Evidence changed: ' + r['file'])
    for r in m['published_runtime_code']:
        if sha(ROOT / r['file']) != r['sha256']:
            raise ValueError('Runtime source differs from frozen release: ' + r['file'])
    archive = DATA / m['source_archive']
    if sha(archive.resolve()) != m['source_archive_sha256']:
        raise ValueError('Frozen archive changed')
    def rows(name):
        with (DATA / name).open() as f:
            return list(csv.DictReader(f))
    common = [r for r in rows('diagnostics/common_gt_counts.csv')
              if r['method'] == 'sam_amg_siglip' and float(r['iou_threshold']) == .5]
    aggregate = {r['aggregation']: float(r['mcmr'])
                 for r in rows('derived/sam_siglip_mcmr_aggregation.csv')}
    if abs(aggregate['equal_target_mean'] - sum(float(r['mcmr']) for r in common)/2) > 1e-12:
        raise ValueError('MCMR aggregation mismatch')
    print(json.dumps({'status': 'passed', 'evidence_files': len(m['files']),
                      'runtime_source_files': len(m['published_runtime_code']),
                      'archive_sha256': m['source_archive_sha256']}, indent=2))

if __name__ == '__main__':
    main()
