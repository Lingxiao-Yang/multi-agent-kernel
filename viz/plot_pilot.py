"""Bar chart of paired speedups from a local GPU pilot run.

Usage: python viz/plot_pilot.py results/local_pilot_v3 reports/local_pilot_v3
Writes pilot.csv, pilot_speedups.png and validation_summary.json to the output dir.
Standalone: needs only numpy and matplotlib (no kernel_study import).
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_json(source / 'pilot_results.json')
    table = []
    for row in rows:
        r = row['result']
        if r['status'] != 'ok':
            table.append({'case': row['case'], 'label': row['label'], 'status': r['status']})
            continue
        table.append({'case': row['case'], 'label': row['label'], 'status': 'ok',
                      'median_ms': r['candidate']['median_ms'],
                      'iqr_ms': r['candidate']['p75_ms'] - r['candidate']['p25_ms'],
                      'paired_speedup': r['paired_speedup'], 'confirmed': r['confirmed'],
                      'library_ms': r['library']['median_ms'],
                      'max_abs_error': max(x['max_abs_error'] for x in r['validation'])})
    keys = sorted(set().union(*(r.keys() for r in table)))
    with (output / 'pilot.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)

    transformed = [r for r in table if r['label'] == 'transformed' and r['status'] == 'ok']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([r['case'] for r in transformed], [r['paired_speedup'] for r in transformed], color='#356f94')
    ax.axvline(1, color='black', linewidth=1)
    ax.set(xlabel='Paired speedup of predefined transformation',
           title='GPU engineering pilot — not an LLM comparison')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'pilot_speedups.png', dpi=180)
    plt.close(fig)

    summary = {'evaluations': len(rows), 'passed': sum(r['status'] == 'ok' for r in table),
               'failed': sum(r['status'] != 'ok' for r in table),
               'hardware': read_json(source / 'manifest.json')['environment'],
               'warning': 'Predefined transforms, laptop GPU, small sample pilot. Do not infer LLM or A100 effects.'}
    (output / 'validation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Wrote', output)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
