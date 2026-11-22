"""Paired per-case effects and clustered bootstrap."""
from __future__ import annotations
import numpy as np
from .core import VARIANTS


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
