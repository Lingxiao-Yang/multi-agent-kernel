"""Schedules and shared starting evidence for paired local-role searches."""
import fcntl
import json
import sqlite3
from contextlib import closing

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


def integrity_problem(directory, rounds):
    ledger = directory/'usage.sqlite3'
    if not ledger.is_file():
        return 'missing per-search ledger'
    try:
        summary = read_json(directory/'summary.json')
        history = summary['history']
        if summary['status']!='complete' or len(history)!=rounds:
            return 'incomplete candidate history'
        for entry in history:
            result = entry['result']
            if result['status'] in ('storage_error','ledger_error','worker_crash') or (
                    result['status']=='proposal_error' and result.get('kind')!='model_output'):
                return 'candidate history contains a search error'
        with (directory/'model_events.jsonl').open(encoding='utf-8') as stream:
            events = list(map(json.loads,stream))
        if any(e['event']=='request_error' for e in events):
            return 'model trace contains a request error'
        requests = [(e['id'],e['stage']) for e in events if e['event']=='request']
        replies = [e for e in events if e['event']=='response']
        stages = ['analyze','advise','implement','review'] * rounds
        if ([stage for _,stage in requests]!=stages or len({i for i,_ in requests})!=len(stages) or
                [(e['id'],e['stage']) for e in replies]!=requests):
            return 'model requests and replies do not match the registered stages'
        expected = []
        for reply in replies:
            usage = reply['metadata']['usage']
            expected.append((directory.name,reply['id'],reply['stage'],usage['prompt_tokens'],usage['completion_tokens']))
        with closing(sqlite3.connect(ledger.resolve().as_uri()+'?mode=ro', uri=True)) as db:
            settled = db.execute('''SELECT job,id,stage,input_tokens,output_tokens,completed,committed
                                    FROM calls ORDER BY id''').fetchall()
        if ([row[:5] for row in settled]!=expected or
                any(completed is None or completed!=committed for *_,completed,committed in settled)):
            return 'model replies and ledger settlements do not match'
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as exc:
        return f'cannot read search accounting: {exc}'
    return None


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
