"""Summarise retained P0 thread-sensitivity runs without refitting models."""
import csv
import json
from pathlib import Path

import numpy  # noqa: F401 - load the backend being recorded
import scipy.linalg  # noqa: F401
from threadpoolctl import threadpool_info

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = ROOT / 'results/submission_v2'
    runs = []
    for name in ['p0', 'p0_threads1', 'p0_threads16']:
        folder = results / name
        summary = json.loads((folder / 'reproduction_summary.json').read_text())
        with (folder / 'metric_checks.csv').open(newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        runs.append({
            'path': str(folder.relative_to(ROOT)),
            'threads': summary['threads'],
            'comparisons': len(rows),
            'matches': summary['matches'],
            'max_absolute_metric_difference': max(abs(float(r['difference'])) for r in rows),
            'all_equal_at_three_decimal_places': all(
                round(float(r['recomputed']), 3) == round(float(r['archived']), 3) for r in rows),
        })
    output = {
        'verified_legacy_reproduction': 'results/submission_v2/p0_threads16',
        'runs': runs,
        'current_native_backend': threadpool_info(),
        'interpretation': 'Thread-sensitive refits; 16 threads recovered archived metrics within unchanged tolerances. '
                          'This does not establish bitwise reproducibility on other platforms or prove historical thread settings.',
    }
    destination = ROOT / 'manifests/submission_v2/p0/numerical_diagnostics.json'
    destination.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(runs, indent=2))


if __name__ == '__main__':
    main()
