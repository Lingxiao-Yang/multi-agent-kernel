"""Paired per-case effects, clustered bootstrap and checkpoint curves."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .core import VARIANTS, read_json, write_json


def geometric_mean(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError('geometric mean needs positive finite observations')
    return float(np.exp(np.mean(np.log(values))))


def paired_effects(rows, resamples=5000, seed=42):
    groups = {}
    for r in rows:
        key = (r['case'], r['seed'])
        if r['variant'] in groups.setdefault(key, {}):
            raise ValueError('duplicate case/seed/variant; do not pool experiments')
        groups[key][r['variant']] = r['speedup']
    complete = {key:v for key,v in groups.items() if set(v) == set(VARIANTS)}
    if not complete:
        return {'complete_pairs':0,'effects':{},'note':'Four-arm matched observations required.'}
    per_case = {}
    for (case, run_seed), v in complete.items():
        logs = {k:np.log(x) for k,x in v.items()}
        effects = [logs['multi']-logs['single'],
                   logs['single_knowledge']-logs['single'],
                   logs['multi_knowledge']-logs['single_knowledge'],
                   logs['multi_knowledge']-logs['multi']-logs['single_knowledge']+logs['single']]
        per_case.setdefault(case, []).append(effects)
    # Average paired seed effects within each case; resample whole cases.
    matrix = np.array([np.mean(v, axis=0) for v in per_case.values()])
    rng = np.random.default_rng(seed)
    boots = matrix[rng.integers(0,len(matrix),(resamples,len(matrix)))].mean(axis=1)
    names = ['roles_without_knowledge','knowledge_single','roles_with_knowledge','interaction_ratio']
    return {'complete_pairs':len(complete),'case_clusters':len(matrix),'bootstrap_seed':seed,
            'resamples':resamples,'effects':{name:{'ratio':float(np.exp(matrix[:,i].mean())),
            'ci95':([float(x) for x in np.exp(np.quantile(boots[:,i],[.025,.975]))] if len(matrix)>1 else None)} for i,name in enumerate(names)},
            'note':('Case-cluster bootstrap over paired log ratios; small-suite inference only.' if len(matrix)>1
                    else 'Only one case cluster: confidence intervals are not estimable.')}


def checkpoint_curve(summary, axis, checkpoints):
    baseline = read_json(Path(summary['_path']).parent/'baseline.result.json')['candidate']['median_ms']
    observations = [(0,1.)]
    best = 1.
    for row in summary['history']:
        b = row.get('budget_after_review',row['budget'])
        position = row['attempt'] if axis == 'attempts' else b[axis]
        if row['accepted']:
            # Search curve uses original baseline latency; final endpoint is separately remeasured.
            best = max(best, baseline/row['result']['candidate']['median_ms'])
        observations.append((position,best))
    return [max([v for p,v in observations if p <= cp],default=1.) for cp in checkpoints]


def validate_cohort(summaries):
    # Source-v1 exposes one family at a time, so different families necessarily
    # have different source hashes. Each family's baseline must remain frozen.
    fingerprints = {(s['manifest']['protocol_hash'],s['manifest']['knowledge_hash'],
                     s['manifest']['provider'],s['manifest']['model'],s['resolved_model'],
                     s['manifest']['environment'].get('gpu')) for s in summaries}
    baselines, case_definitions = {}, {}
    for s in summaries:
        m=s['manifest']
        baselines.setdefault(m['case']['family'],set()).add(m['baseline_hash'])
        case_definitions.setdefault(m['case']['id'],set()).add(m['case_hash'])
    if len(fingerprints)!=1 or any(len(x)!=1 for x in [*baselines.values(),*case_definitions.values()]):
        raise ValueError('Incompatible experiment manifests; analyze cohorts separately')


def analyze(input_dir, output_dir, allow_controls=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    rows, summaries, excluded = [], [], []
    for path in sorted(Path(input_dir).rglob('summary.json')):
        s = read_json(path)
        if s.get('status') != 'complete':
            excluded.append({'path':str(path),'reason':s.get('status')})
            continue
        if s.get('stop_reason') == 'token_safety_cap':
            excluded.append({'path':str(path),'reason':'token_limited_report_separately','speedup':s.get('speedup')})
            continue
        if s.get('stop_reason') not in ['candidate_cap','call_cap']:
            excluded.append({'path':str(path),'reason':'interrupted_search','detail':s.get('stop_reason')})
            continue
        if not s.get('scientific_evidence') and not allow_controls:
            continue
        m = s['manifest']
        if m.get('drop_role') or m.get('drop_category'):
            continue
        s['_path'] = str(path)
        summaries.append(s)
        rows.append({'case':m['case']['id'],'family':m['case']['family'],'variant':m['variant'],'seed':m['seed'],
                     'provider':m['provider'],'gpu':m['environment'].get('gpu','unknown'), 'speedup':s['speedup'],
                     'attempts':s['budget']['attempts'],'calls':s['budget']['calls'],'tokens':s['budget']['tokens'],
                     'valid_attempts':sum(x['result']['status']=='ok' for x in s['history']),
                     'retained_attempts':sum(x['accepted'] for x in s['history']), 'wall_seconds':s['wall_seconds'],
                     'estimated_usd':s.get('estimated_inference_usd'),
                     'library_ms':s['final']['result']['library']['median_ms'],
                     'candidate_ms':s['final']['result']['candidate']['median_ms'],
                     'stop_reason':s['stop_reason']})
    if not rows:
        raise ValueError('No eligible complete runs. Scripted controls require --allow-controls.')
    # Refuse accidental mixing of hardware, source, protocol, model or provider.
    validate_cohort(summaries)
    with (output_dir/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    report = {'n_runs':len(rows), 'evidence_type':'engineering-control' if allow_controls else 'LLM-study',
              'variant_geomeans':{v:geometric_mean([r['speedup'] for r in rows if r['variant']==v]) for v in VARIANTS if any(r['variant']==v for r in rows)},
              'paired':paired_effects(rows), 'excluded_runs':excluded,
              'excluded_incomplete_runs':len(list(Path(input_dir).rglob('summary.json')))-len(rows)}
    write_json(output_dir/'aggregate.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,3.8))
    for ax, axis in zip(axes,['attempts','calls','tokens']):
        top = max(s['budget'][axis] for s in summaries)
        checkpoints = np.linspace(0,max(top,1),21)
        for variant in VARIANTS:
            curves = [checkpoint_curve(s,axis,checkpoints) for s in summaries if s['manifest']['variant']==variant]
            if curves:
                ax.plot(checkpoints,np.exp(np.log(curves).mean(axis=0)),label=variant)
        ax.set(xlabel=axis,ylabel='Geometric mean best search speedup')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle('Engineering controls (not LLM evidence)' if allow_controls else 'Budget-matched search curves')
    fig.tight_layout(); fig.savefig(output_dir/'search_curves.png',dpi=180); plt.close(fig)
    return report
