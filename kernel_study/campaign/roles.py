"""Schedules and shared starting evidence for paired local-role searches."""
import fcntl

from ..judge.evaluate import ncu_evidence
from ..records import digest, read_json, write_json


def registration(output, plan):
    spec = read_json(output/'role-study.json')
    cases, seeds = spec['cases'], spec['seeds']
    if not cases or len(set(cases))!=len(cases) or any(type(c) is not int or str(c) not in plan['cases'] for c in cases):
        raise ValueError('register distinct public case IDs')
    if not seeds or len(set(seeds))!=len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('register distinct integer search seeds')
    if set(seeds).intersection(plan['development_seeds']):
        raise ValueError('qualification seeds must be reported separately')
    if type(spec['knowledge']) is not bool:
        raise ValueError('register whether to include the knowledge-card variants')
    return spec


def comparisons(spec):
    pairs = [('single','multi')]
    if spec['knowledge']:
        pairs.extend([('single_knowledge','multi_knowledge'),('single','single_knowledge'),('multi','multi_knowledge')])
    return pairs


def schedule(spec):
    jobs = []
    for i,case in enumerate(spec['cases']):
        for j,seed in enumerate(spec['seeds']):
            arms = ['single','multi']
            if (i+j)%2:
                arms.reverse()
            if spec['knowledge']:
                cards = [a+'_knowledge' for a in arms]
                arms = cards+arms if (i+j//2)%2 else arms+cards
            jobs.extend((case,a,seed) for a in arms)
    return jobs


def starting_metrics(pair, source, problem):
    path = pair/'starting-metrics.json'
    fingerprint = digest(source.read_text())
    with (pair/'starting-metrics.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not path.exists():
            evidence = ncu_evidence(source,problem,pair/'initial-ncu')
            write_json(path,dict(source_hash=fingerprint,metrics=evidence['metrics']))
        record = read_json(path)
    if record['source_hash']!=fingerprint:
        raise ValueError('initial hardware evidence belongs to a different starting program')
    return record['metrics']
