"""Summarize independently checked candidates at matched attempt and dollar budgets."""
import argparse
import ast
import json
import math
from pathlib import Path

from .core import read_json, write_json
from .public_cases import experiment
from .search import run_worker


def changed_program(source, start):
    def statements(text):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and ast.get_docstring(node) is not None:
                node.body.pop(0)
        return ast.dump(tree)
    return statements(source)!=statements(start)


def audit(output, agent):
    summary = read_json(output/'summary.json')
    manifest, final = summary['manifest'],summary['final']
    case,protocol = manifest['case'],manifest['protocol']
    log = output/'model_events.jsonl'
    events = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    replies = {e['id']:e['budget'] for e in events if e['event']=='response'}
    proposals = [e for e in events if e['event']=='request' and e['stage'] in ('implement','proposal','repair')]
    history = {h['attempt']:h for h in summary['history']}
    start = output/'baseline.py'
    start_config = read_json(output/'baseline.request.json')['config'] if agent=='single' else None
    candidates = []
    for call in proposals:
        budget = replies.get(call['id'],call['budget'])
        i = budget['attempts']
        h = history.get(i,{})
        point = dict(attempt=i,usd=budget['committed_usd'],status=h.get('result',{}).get('status','no_submission'))
        if point['status']=='ok':
            source = output/(f'candidate_{i:03}.py' if agent=='single' else f'proposal_{i:03}.py')
            if changed_program(source.read_text(),start.read_text()) or (agent=='single' and h.get('config')!=start_config):
                req = dict(case=case,protocol=protocol,source=str(source),config=h.get('config'),
                           incumbent_source=str(start),incumbent_config=start_config,final=True,
                           interface='declarative' if agent=='single' else 'native')
                checked = run_worker(req,output,f'audit_{i:03}')
                point['status'] = checked['status']
                if checked['status']=='ok': point['latency_ms'] = checked['candidate']['median_ms']
        candidates.append(point)
    row = dict(agent=agent,case=case['public_id'],seed=manifest['seed'],status=summary['status'],
               correct_new=any('latency_ms' in h for h in candidates),candidates=candidates,
               budget=summary['budget'],cost_usd=summary['estimated_inference_usd'],wall_seconds=summary['wall_seconds'])
    workers = [read_json(p) for p in output.glob('*.result.json')]
    row['evaluator_process_seconds'] = sum(w.get('process_seconds',0.) for w in workers)
    row['gpu_slot_seconds'] = summary['wall_seconds'] + sum(
        read_json(p).get('process_seconds',0.) for p in output.glob('audit_*.result.json'))
    if final:
        r = final['result']
        row.update(latency_ms=r['candidate']['median_ms'],fallback=final['candidate_id']==0,
                   initial_ms=r['incumbent']['median_ms'],speedup=r['paired_speedup'],
                   compile_speedup=r['compile']['median_ms']/r['compile_candidate']['median_ms'])
    write_json(output/'audit.json',row)
    return row


def geometric(values):
    return math.exp(sum(math.log(v) for v in values)/len(values))


def compare(rows, cases, seeds):
    keyed = {(r['case'],r['seed'],r['agent']):r for r in rows}
    if len(keyed)!=len(rows): raise ValueError('duplicate search')
    tasks, ratios, valid = {},[],{'single':0,'greedy':0}
    fallbacks = {'single':0,'greedy':0}
    paired = 0
    for case in cases:
        values = []
        for seed in seeds:
            pair = [keyed.get((case,seed,a)) for a in valid]
            if any(not r or r['status']!='complete' or 'latency_ms' not in r for r in pair): continue
            local,greedy = pair
            values.append(local['latency_ms']/greedy['latency_ms'])
            for r in pair:
                valid[r['agent']] += r['correct_new']
                fallbacks[r['agent']] += r['fallback']
        paired += len(values)
        tasks[str(case)] = {'paired_repeats':len(values),'ratios':values,'ratio':geometric(values) if values else None}
        if values: ratios.append(geometric(values))
    complete = paired==len(cases)*len(seeds)
    ratio = geometric(ratios) if ratios else None
    if not complete: decision = 'incomplete'
    elif min(fallbacks.values()) == paired: decision = 'pending-all-fallbacks'
    elif ratio>1.10 and abs(fallbacks['single']-fallbacks['greedy'])==1: decision = 'pending-fallback-remeasurement'
    elif valid['single']>=valid['greedy'] and ratio<=1.10: decision = 'passes'
    else: decision = 'below-criterion'
    return dict(complete=complete,paired_searches=paired,tasks=tasks,ratio=ratio,correct_new=valid,
                fallbacks=fallbacks,decision=decision)


def prefix(row, axis, limit):
    best = row['initial_ms']
    for c in row['candidates']:
        if c[axis]<=limit and 'latency_ms' in c:
            best = min(best,c['latency_ms'])
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['development','holdout'])
    parser.add_argument('output',type=Path)
    parser.add_argument('--freeze',action='store_true',help='Register the measured single implementation before holdout')
    args = parser.parse_args()
    plan = experiment()
    rows = [read_json(p) for p in sorted(args.output.glob(args.phase+'-*/audit.json'))]
    report = compare(rows,plan[args.phase],plan[args.phase+'_seeds'])
    report['searches'] = rows
    budgets = []
    for case in plan[args.phase]:
        for seed in plan[args.phase+'_seeds']:
            pair = [r for r in rows if r['case']==case and r['seed']==seed and 'initial_ms' in r]
            if len(pair)!=2: continue
            for axis,limit in [('attempt',min(r['budget']['attempts'] for r in pair)),('usd',min(r['cost_usd'] for r in pair))]:
                points = sorted({c[axis] for r in pair for c in r['candidates'] if c[axis]<=limit}|{limit})
                budgets.extend(dict(case=case,seed=seed,axis=axis,limit=p,
                                    **{r['agent']:prefix(r,axis,p) for r in pair}) for p in points)
    report['matched_budget_ms'] = budgets
    write_json(args.output/(args.phase+'-report.json'),report)
    if args.freeze:
        if args.phase!='development' or not report['complete']:
            parser.error('complete development measurements are required to freeze single')
        from .agent_campaign import code_hashes
        measured = read_json(args.output/'development-protocol.json')
        if code_hashes()!=measured['code_hashes']:
            parser.error('development measurements belong to different code')
        write_json(args.output/'strategy_freeze.json',{'code_hashes':measured['code_hashes'],
                   'revision':measured['revision'],'development_decision':report['decision']})
    print(report['decision'],report['paired_searches'])


if __name__=='__main__': main()
