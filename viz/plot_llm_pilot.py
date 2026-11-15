"""Per-case dot plot of final speedup by variant (one dot per seed, tick = geometric mean).

Usage: python viz/plot_llm_pilot.py results/a100_llm_pilot_3582416/analysis/runs.csv out.png
Standalone: needs only numpy and matplotlib.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = ['single', 'single_knowledge', 'multi', 'multi_knowledge']
LABELS = {'single': 'Single', 'single_knowledge': 'Single + knowledge',
          'multi': 'Multi-agent', 'multi_knowledge': 'Multi-agent + knowledge'}
# Validated categorical slots 1-4 (blue, orange, aqua, yellow); identity is also on the axis.
COLORS = dict(zip(VARIANTS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']))
SURFACE, INK, MUTED, GRID = '#fcfcfb', '#1f1f1e', '#6b6a63', '#e4e3df'


def main(runs_csv, output):
    data = defaultdict(lambda: defaultdict(list))
    gpu = ''
    with open(runs_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            data[row['case']][row['variant']].append(float(row['speedup']))
            gpu = row['gpu']
    cases = sorted(data)
    xmax = max(v for c in cases for vs in data[c].values() for v in vs) * 1.4

    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 3.4), sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    for ax, case in zip(axes, cases):
        ax.set_facecolor(SURFACE)
        for y, variant in enumerate(VARIANTS):
            vals = data[case][variant]
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter(vals, y + np.asarray(jitter), s=64, color=COLORS[variant],
                       edgecolor=SURFACE, linewidth=2, zorder=3)
            g = float(np.exp(np.mean(np.log(vals))))
            ax.plot([g, g], [y - 0.28, y + 0.28], color=INK, linewidth=2, zorder=4)
            ax.text(xmax / 1.02, y, f'{g:.2f}×', va='center', ha='right', fontsize=9, color=INK)
        ax.axvline(1, color=MUTED, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_xscale('log')
        ax.set_xlim(0.85, xmax)
        ticks = [t for t in [1, 1.5, 2, 3, 5, 8] if t < xmax / 1.1]
        ax.set_xticks(ticks, [f'{t:g}×' for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(range(len(VARIANTS)), [LABELS[v] for v in VARIANTS])
        ax.set_ylim(len(VARIANTS) - 0.5, -0.5)
        ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        for lab in ax.get_yticklabels():
            lab.set_color(INK)
        ax.set_title(case, loc='left', fontsize=11, color=INK)
        ax.set_xlabel('Final speedup vs. baseline (log scale)', color=MUTED, fontsize=9)
    fig.suptitle(f'A100 LLM pilot — final speedup per seed (dots) and geometric mean (tick, value at right)\n'
                 f'{gpu}; 2 candidate attempts per search; pilot evidence only',
                 x=0.01, ha='left', fontsize=10, color=INK)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=SURFACE)
    plt.close(fig)
    print('Wrote', output)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
