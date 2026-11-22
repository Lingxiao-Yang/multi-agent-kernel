"""Summarize independently checked candidates at matched attempt and dollar budgets."""

from ..judge.select import measure_pair
from ..records import code_hashes, read_json, write_json
from ..tasks.kernelbench import experiment
from .factorial import geometric_mean
from .jobs import job_name, latest_attempt
from .preflight import check_device


def compare(rows, cases, seeds, measurements):
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
            measured = measurements.get((case,seed), {})
            if measured.get('status')!='ok': continue
            values.append(measured['candidate']['median_ms']/measured['incumbent']['median_ms'])
            for r in pair:
                valid[r['agent']] += r['correct_new']
                fallbacks[r['agent']] += r['fallback']
        paired += len(values)
        tasks[str(case)] = {'paired_repeats':len(values),'ratios':values,'ratio':geometric_mean(values) if values else None}
        if values: ratios.append(geometric_mean(values))
    complete = paired==len(cases)*len(seeds)
    ratio = geometric_mean(ratios) if ratios else None
    if not complete: decision = 'incomplete'
    elif min(fallbacks.values()) == paired: decision = 'pending-all-fallbacks'
    elif ratio>1.10 and abs(fallbacks['single']-fallbacks['greedy'])==1: decision = 'pending-fallback-remeasurement'
    elif valid['single']>=valid['greedy'] and ratio<=1.10: decision = 'passes'
    else: decision = 'below-criterion'
    return dict(complete=complete,paired_searches=paired,tasks=tasks,ratio=ratio,correct_new=valid,
                fallbacks=fallbacks,decision=decision)


def prefix(row, axis, limit):
    best = row['initial_ms']
    for c in row['matched_candidates']:
        if c[axis]<=limit and c.get('selected'):
            best = c['latency_ms']
    return best


def phase_rows(root, phase, cases, seeds):
    rows, attempts, measurements = [], [], {}
    for case in cases:
        for seed in seeds:
            result = root/'pairs'/f'{phase}-{case}-{seed}'/'direct.result.json'
            if result.exists():
                measurements[case,seed] = read_json(result)
            for agent in ('single','greedy'):
                name = job_name(phase,case,agent,seed)
                latest = latest_attempt(root,name)
                prior = root/name
                records = []
                for directory in ([prior,latest] if prior!=latest else [prior]):
                    path = directory/'audit.json'
                    if not path.exists():
                        continue
                    row = read_json(path)
                    records.append(row)
                    attempts.append(dict(directory=directory.name, status=row['status'], reason=row.get('stop_reason'),
                                         completed_usd=row['cost_usd'], committed_usd=row.get('committed_usd',0.),
                                         budget=row['budget']))
                    if directory==latest:
                        row = dict(row, run=directory.name)
                        previous_cost = sum(item.get('committed_usd',0.) for item in records[:-1])
                        row['total_committed_usd'] = previous_cost + row.get('committed_usd',0.)
                        row['total_completed_usd'] = sum(item['cost_usd'] for item in records)
                        previous_attempts = sum(item['budget']['attempts'] for item in records[:-1])
                        row['total_attempts'] = previous_attempts + row['budget']['attempts']
                        row['matched_candidates'] = [dict(c, usd=previous_cost+c['usd'],attempt=previous_attempts+c['attempt'])
                                                     for c in row['candidates']]
                        rows.append(row)
    return rows,attempts,measurements


def run_report(args, parser):
    args.output = args.output.resolve()
    plan = experiment()
    if args.measure_pairs:
        device = check_device(plan)
        if read_json(args.output/(args.phase+'-protocol.json'))['code_hashes'] != code_hashes():
            parser.error('pair measurements require the registered evaluation code')
        for case in plan[args.phase]:
            for seed in plan[args.phase+'_seeds']:
                pair = args.output/'pairs'/f'{args.phase}-{case}-{seed}'
                assigned = pair/'device.json'
                if assigned.exists() and read_json(assigned)['uuid'] != device['uuid']:
                    continue
                sides = [latest_attempt(args.output,job_name(args.phase,case,agent,seed)) for agent in ('single','greedy')]
                measure_pair(pair,*sides)
    rows,attempts,measurements = phase_rows(args.output,args.phase,plan[args.phase],plan[args.phase+'_seeds'])
    report = compare(rows,plan[args.phase],plan[args.phase+'_seeds'],measurements)
    report['attempts'] = attempts
    report['direct_pairs'] = {f'{case}-{seed}':data for (case,seed),data in measurements.items()}
    report['searches'] = rows
    budgets = []
    for case in plan[args.phase]:
        for seed in plan[args.phase+'_seeds']:
            pair = [r for r in rows if r['case']==case and r['seed']==seed and 'initial_ms' in r]
            if len(pair)!=2: continue
            for axis,limit in [('attempt',min(r['total_attempts'] for r in pair)),('usd',min(r['total_committed_usd'] for r in pair))]:
                points = sorted({c[axis] for r in pair for c in r['matched_candidates'] if c[axis]<=limit}|{limit})
                budgets.extend(dict(case=case,seed=seed,axis=axis,limit=p,
                                    **{r['agent']:prefix(r,axis,p) for r in pair}) for p in points)
    report['matched_budget_ms'] = budgets
    write_json(args.output/(args.phase+'-report.json'),report)
    if args.freeze:
        if args.phase!='development' or not report['complete']:
            parser.error('complete development measurements are required to freeze single')
        measured = read_json(args.output/'development-protocol.json')
        if code_hashes()!=measured['code_hashes']:
            parser.error('development measurements belong to different code')
        write_json(args.output/'strategy_freeze.json',{'code_hashes':measured['code_hashes'],
                   'revision':measured['revision'],'development_decision':report['decision']})
    print(report['decision'],report['paired_searches'])
