"""Best-so-far speedup vs. budget (attempts / calls / tokens) for each variant.

Usage: python viz/plot_search_curves.py results/gpu_control_matrix out.png [--controls]
Reads every complete summary.json under the input dir. Plotting logic mirrors
kernel_study.analysis on main; standalone (numpy + matplotlib only).
Pass --controls to include scripted engineering-control runs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = ('single', 'single_knowledge', 'multi', 'multi_knowledge')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checkpoint_curve(summary, axis, checkpoints):
    baseline = read_json(Path(summary['_path']).parent / 'baseline.result.json')['candidate']['median_ms']
    observations, best = [(0, 1.)], 1.
    for row in summary['history']:
        b = row.get('budget_after_review', row['budget'])
        position = row['attempt'] if axis == 'attempts' else b[axis]
        if row['accepted']:
            best = max(best, baseline / row['result']['candidate']['median_ms'])
        observations.append((position, best))
    return [max([v for p, v in observations if p <= cp], default=1.) for cp in checkpoints]


def load_summaries(input_dir, allow_controls):
    out = []
    for path in sorted(Path(input_dir).rglob('summary.json')):
        s = read_json(path)
        if s.get('status') != 'complete' or s.get('stop_reason') not in ('candidate_cap', 'call_cap'):
            continue
        if not s.get('scientific_evidence') and not allow_controls:
            continue
        m = s['manifest']
        if m.get('drop_role') or m.get('drop_category'):
            continue
        s['_path'] = str(path)
        out.append(s)
    return out


def main(input_dir, output, allow_controls):
    summaries = load_summaries(input_dir, allow_controls)
    if not summaries:
        raise SystemExit('No eligible complete runs (scripted controls need --controls).')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, axis in zip(axes, ['attempts', 'calls', 'tokens']):
        top = max(s['budget'][axis] for s in summaries)
        checkpoints = np.linspace(0, max(top, 1), 21)
        for variant in VARIANTS:
            curves = [checkpoint_curve(s, axis, checkpoints) for s in summaries
                      if s['manifest']['variant'] == variant]
            if curves:
                ax.plot(checkpoints, np.exp(np.log(curves).mean(axis=0)), label=variant)
        ax.set(xlabel=axis, ylabel='Geometric mean best search speedup')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Engineering controls (not LLM evidence)' if allow_controls else 'Budget-matched search curves')
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print('Wrote', output)


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--controls']
    if len(args) != 2:
        raise SystemExit(__doc__)
    main(args[0], args[1], '--controls' in sys.argv)
