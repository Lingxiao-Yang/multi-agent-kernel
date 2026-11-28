"""Summarize independently checked candidates at matched attempt and dollar budgets."""
from math import comb
from pathlib import Path

from ..judge.select import measure_pair
from ..records import code_hashes, digest, read_json, write_json
from ..tasks.kernelbench import experiment
from .factorial import geometric_mean
from .jobs import campaign_plan, job_name, latest_attempt, measure_finalists, phase_budget
from .launch_sweep import matvec_label
from .preflight import check_device
from . import roles


def rejection_counts(candidates):
    counts = dict.fromkeys(('static_check','compilation','numerical','timeout','unchanged_incumbent','unconfirmed','confirmed'),0)
    categories = {'static_rejected':'static_check', 'compile_error':'compilation',
                  'incorrect':'numerical', 'timeout':'timeout', 'unchanged_incumbent':'unchanged_incumbent',
                  'unconfirmed':'unconfirmed','confirmed':'confirmed'}
    for candidate in candidates:
        status = candidate.get('endpoint_status') or candidate['status']
        if status=='ok' and 'selected' in candidate:
            status = 'confirmed' if candidate['selected'] else 'unconfirmed'
        if status in categories:
            counts[categories[status]] += 1
    return counts


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
    rejections = {agent:rejection_counts([]) for agent in valid}
    for row in rows:
        counts = row['rejections'] if 'rejections' in row else rejection_counts(row.get('candidates',[]))
        for category,count in counts.items():
            rejections[row['agent']][category] += count
    return dict(rejections=rejections,complete=complete,paired_searches=paired,tasks=tasks,ratio=ratio,correct_new=valid,
                fallbacks=fallbacks,decision=decision)


def prefix(row, axis, limit):
    best = row['initial_ms']
    for c in row['matched_candidates']:
        if c[axis]<=limit and c.get('selected'):
            best = c['latency_ms']
    return best


def verifier_timeouts(path):
    if not path.is_file():
        return None
    return sum(line.endswith(' - ERROR - Test timed out') for line in path.read_text().splitlines())


def parent_comparison(directory, attempt, previous):
    request = directory/f'audit_{attempt:03d}.request.json'
    result = directory/f'audit_{attempt:03d}.result.json'
    if not request.is_file() or not result.is_file():
        return None
    judged = read_json(result)
    if Path(read_json(request)['incumbent_source']).name != previous or judged['status'] != 'ok':
        return None
    return dict(previous=previous, confirmed=judged['confirmed'], candidate_ms=judged['candidate']['median_ms'],
                parent_ms=judged['incumbent']['median_ms'])


def round_benchmarks(artifacts, proposals):
    measured = []
    bench = artifacts/'benchmark_results.json'
    if bench.is_file():
        for name, timing in read_json(bench)['kernels'].items():
            kernel = artifacts/(name+'.py')
            if name == 'kernel_round_0' or not kernel.is_file(): continue
            entry = proposals.get(digest(kernel.read_text()))
            if entry:
                public = entry['result'].get('candidate',{})
                measured.append(dict(attempt=entry['attempt'],upstream_ms=timing['time_ms'],public_ms=public.get('median_ms')))
    return measured


def greedy_trajectory(directory, summary):
    upstream = directory/'greedy'
    proposals = {}
    for entry in summary['history']:
        key = entry['source_hash']
        proposals[key] = None if key in proposals else entry
    baseline = directory/'baseline.py'
    baseline_hash = digest(baseline.read_text()) if baseline.is_file() else None
    previous = 'baseline.py'
    rounds = []
    for step in summary['rounds']:
        worker = upstream/'workers/w0'/f'r{step["round"]}'
        artifacts = worker/'artifacts'
        parent_file = artifacts/'kernel_round_0.py'
        parent = change = None
        if parent_file.is_file():
            source_hash = digest(parent_file.read_text())
            if source_hash == baseline_hash:
                parent = 'baseline.py'
            elif entry := proposals.get(source_hash):
                attempt = entry['attempt']
                parent = f'proposal_{attempt:03d}.py'
                if parent != previous:
                    change = parent_comparison(directory, attempt, previous)
        rounds.append(dict(**step,parent=parent,parent_change=change,upstream_candidates=round_benchmarks(artifacts,proposals),
                           upstream_candidate_timeouts=verifier_timeouts(worker/'logs/worker_0.log')))
        previous = parent
    return dict(upstream_initial_timeouts=verifier_timeouts(upstream/'initial_verify/worker_-1.log'),rounds=rounds)


def phase_rows(root, phase, cases, seeds, agents=('single','greedy'), check_complete=None):
    rows, attempts, measurements = [], [], {}
    for case in cases:
        for seed in seeds:
            result = root/'pairs'/f'{phase}-{case}-{seed}'/'direct.result.json'
            if result.exists():
                measurements[case,seed] = read_json(result)
            for agent in agents:
                name = job_name(phase,case,agent,seed)
                latest = latest_attempt(root,name)
                prior = root/name
                records = []
                for directory in ([prior,latest] if prior!=latest else [prior]):
                    path = directory/'audit.json'
                    if not path.exists():
                        continue
                    row = read_json(path)
                    if check_complete and row['status']=='complete':
                        reason = check_complete(directory)
                        if reason:
                            row.update(status='invalid',stop_reason=reason)
                    records.append(row)
                    attempts.append(dict(directory=directory.name, status=row['status'], reason=row.get('stop_reason'),
                                         completed_usd=row['cost_usd'], committed_usd=row.get('committed_usd',0.),
                                         budget=row['budget']))
                    if directory==latest:
                        row = dict(row, run=directory.name)
                        if agent=='greedy' and (directory/'summary.json').is_file():
                            row['upstream_trajectory'] = greedy_trajectory(directory,read_json(directory/'summary.json'))
                        previous_cost = sum(item.get('committed_usd',0.) for item in records[:-1])
                        row['total_committed_usd'] = previous_cost + row.get('committed_usd',0.)
                        row['total_completed_usd'] = sum(item['cost_usd'] for item in records)
                        previous_attempts = sum(item['budget']['attempts'] for item in records[:-1])
                        row['total_attempts'] = previous_attempts + row['budget']['attempts']
                        row['rejections'] = rejection_counts(c for run in records for c in run['candidates'])
                        row['matched_candidates'] = [dict(c, usd=previous_cost+c['usd'],attempt=previous_attempts+c['attempt'])
                                                     for c in row['candidates']]
                        rows.append(row)
    return rows,attempts,measurements


def matvec_screen_times(directory, case):
    manifest = read_json(directory/'manifest.json')
    if manifest['case'] != case:
        raise ValueError('launch sweep belongs to a different task')
    times = []
    for config in manifest['grid']:
        row = read_json(directory/f'screen-{matvec_label(config)}.result.json')
        times.append(row['candidate']['median_ms'] if row['status']=='ok' else float('inf'))
    return times


def matvec_endpoint(ms, sweep):
    result = dict(latency_ms=ms)
    if sweep is not None:
        slower = sum(t > ms for t in sweep)
        result['random_four_at_or_below'] = 1 - comb(slower,4)/comb(len(sweep),4)
    return result


def report_matvec_recheck(args, plan, parser):
    seeds = plan['development_seeds']
    if args.freeze or args.measure_pairs or (args.require_seed is not None and args.require_seed not in seeds):
        parser.error('matvec rechecks report individual endpoints for the registered seeds')
    rows,attempts,_ = phase_rows(args.output,args.phase,[4],seeds)
    results = {}
    reference = plan['launch_reference_ms']['4']
    sweep = matvec_screen_times(args.matvec_sweep,plan['cases']['4']) if args.matvec_sweep else None
    for row in rows:
        path = args.output/row['run']/'matched-evidence.json'
        proof = read_json(path) if path.is_file() else None
        row['matched_evidence'] = proof
        if row['status']!='complete' or 'latency_ms' not in row or not (proof and proof['identical']):
            continue
        result = matvec_endpoint(row['latency_ms'],sweep)
        result.update(at_or_below_random_median=row['latency_ms']<=reference['random_best_of_four'],
                      cost_usd=row['total_completed_usd'],previous={})
        for agent in ('greedy','single'):
            run = job_name('matvec-development',4,agent,row['seed'])
            prior = latest_attempt(args.output,run)/'audit.json'
            if prior.is_file():
                audit = read_json(prior)
                if audit['status']=='complete' and 'latency_ms' in audit:
                    result['previous'][agent] = matvec_endpoint(audit['latency_ms'],sweep)
        if row['seed']==11:
            result['below_3_57_ms'] = row['latency_ms'] < 3.57
        results[str(row['seed'])] = result
    complete = len(results)==len(seeds)
    decision = 'matvec-recheck-complete' if complete else 'incomplete'
    write_json(args.output/(args.phase+'-report.json'),dict(complete=complete,decision=decision,seeds=results,
               reference_ms=reference,random_four_population=len(sweep) if sweep is not None else None,
               attempts=attempts,searches=rows))
    print(decision,results)
    if args.require_seed is not None and str(args.require_seed) not in results:
        raise SystemExit(3)


def row_qualification(report, rows, cases, references, threshold):
    for case in cases:
        task = report['tasks'][str(case)]
        task['reference_ms'] = references[str(case)]
        pair = [r for r in rows if r['case']==case]
        ratio = task['ratio']
        if ratio is None or not any((r.get('matched_evidence') or {}).get('identical') for r in pair):
            outcome = 'incomplete'
        elif all(r['fallback'] for r in pair):
            outcome = 'both-retained-start'
        elif ratio > threshold:
            outcome = 'single-slower'
        elif ratio < 1/threshold:
            outcome = 'single-faster'
        else:
            outcome = 'within-timing-tolerance'
        task['outcome'] = outcome
    report['complete'] = all(task['outcome']!='incomplete' for task in report['tasks'].values())
    if report['complete']:
        lost = any(task['outcome']=='single-slower' for task in report['tasks'].values())
        report['decision'] = 'row-qualification-failed' if lost else 'row-qualification-passed'
    else:
        report['decision'] = 'incomplete'


def matched_budgets(rows, cases, seeds):
    budgets = []
    for case in cases:
        for seed in seeds:
            pair = [r for r in rows if r['case']==case and r['seed']==seed and 'initial_ms' in r]
            if len(pair)!=2: continue
            for axis,limit in [('attempt',min(r['total_attempts'] for r in pair)),('usd',min(r['total_committed_usd'] for r in pair))]:
                points = sorted({c[axis] for r in pair for c in r['matched_candidates'] if c[axis]<=limit}|{limit})
                budgets.extend(dict(case=case,seed=seed,axis=axis,limit=p,
                                    **{r['agent']:prefix(r,axis,p) for r in pair}) for p in points)
    return budgets


def report_roles(root, spec, references, rounds):
    cases,seeds = spec['cases'],spec['seeds']
    contrasts = roles.comparisons(spec)
    agents = list(dict.fromkeys(agent for contrast in contrasts for agent in contrast))
    rows,attempts,_ = phase_rows(root,'roles',cases,seeds,agents,
                                check_complete=lambda directory: roles.integrity_problem(directory,rounds))
    completed = {}
    for row in rows:
        evidence = root/row['run']/'matched-evidence.json'
        if row['status']=='complete' and evidence.is_file() and read_json(evidence)['identical']:
            completed[row['case'],row['seed'],row['agent']] = row
    tasks = {}
    for case in cases:
        tasks[str(case)] = {'reference_ms':references.get(str(case),{})}
        for agent in agents:
            runs = [row for (c,s,a),row in completed.items() if c==case and a==agent]
            for row in runs:
                row['at_or_below_reference'] = {name:row['latency_ms']<=ms for name,ms in tasks[str(case)]['reference_ms'].items()}
            fallbacks = sum(r['fallback'] for r in runs)
            tasks[str(case)][agent] = dict(completed=len(runs),fallbacks=fallbacks,confirmed_gain_searches=len(runs)-fallbacks)
    effects = {}
    for left,right in contrasts:
        name = left+'-vs-'+right
        direct,means = {},{}
        for case in cases:
            ratios = []
            for seed in seeds:
                file = root/'pairs'/f'roles-{case}-{seed}'/name/'direct.result.json'
                if not file.is_file() or not all((case,seed,a) in completed for a in (left,right)):
                    continue
                measure = read_json(file)
                if measure['status']=='ok':
                    direct[f'{case}-{seed}'] = measure
                    ratios.append(measure['candidate']['median_ms']/measure['incumbent']['median_ms'])
            if len(ratios)==len(seeds):
                means[str(case)] = geometric_mean(ratios)
        ratio = geometric_mean(list(means.values())) if len(means)==len(cases) else None
        effects[name] = dict(ratio=ratio,case_ratios=means,direct_pairs=direct,
                             matched_budget_ms=matched_budgets([r for r in completed.values() if r['agent'] in (left,right)],cases,seeds))
    complete = all(e['ratio'] is not None for e in effects.values())
    write_json(root/'roles-report.json',dict(complete=complete,registration=spec,tasks=tasks,
                                            comparisons=effects,searches=rows,attempts=attempts))
    print('complete' if complete else 'incomplete',len(completed))


def run_report(args, parser):
    args.output = args.output.resolve()
    plan = campaign_plan(args.phase,args.output) if args.phase=='roles' else experiment()
    if args.phase=='matvec-recheck':
        return report_matvec_recheck(args,plan,parser)
    if getattr(args,'matvec_sweep',None) is not None:
        parser.error('--matvec-sweep applies to matvec-recheck')
    if getattr(args,'require_seed',None) is not None:
        parser.error('--require-seed applies to matvec-recheck')
    if args.phase == 'matched-smoke':
        phase = plan['phases'][args.phase]
        cases, seeds = [phase['case']], [phase['seed']]
    elif args.phase == 'matvec-development':
        cases, seeds = [4], plan['development_seeds']
    elif args.phase == 'row-qualification':
        spec = plan['row_qualification']
        cases, seeds = spec['cases'], [spec['seed']]
    elif args.phase=='roles':
        if args.freeze:
            parser.error('the role study uses its registered mainline commit')
        cases,seeds = plan['role_study']['cases'],plan['role_study']['seeds']
    else:
        cases, seeds = plan[args.phase], plan[args.phase+'_seeds']
    if args.measure_pairs:
        device = check_device(plan)
        if read_json(args.output/(args.phase+'-protocol.json'))['code_hashes'] != code_hashes():
            parser.error('pair measurements require the registered evaluation code')
        for case in cases:
            for seed in seeds:
                pair = args.output/'pairs'/f'{args.phase}-{case}-{seed}'
                assigned = pair/'device.json'
                if assigned.exists() and read_json(assigned)['uuid'] != device['uuid']:
                    continue
                if args.phase=='roles':
                    if assigned.is_file() and read_json(assigned)['node']==device['node']:
                        measure_finalists(args.output,args.phase,case,seed,plan)
                else:
                    sides = [latest_attempt(args.output,job_name(args.phase,case,agent,seed)) for agent in ('single','greedy')]
                    measure_pair(pair,*sides)
    if args.phase=='roles':
        return report_roles(args.output,plan['role_study'],plan['launch_reference_ms'],phase_budget(plan,'roles')['rounds'])
    rows,attempts,measurements = phase_rows(args.output,args.phase,cases,seeds)
    report = compare(rows,cases,seeds,measurements)
    if args.phase == 'row-qualification':
        for row in rows:
            if row['agent']=='single':
                path = args.output/row['run']/'matched-evidence.json'
                row['matched_evidence'] = read_json(path) if path.is_file() else None
        row_qualification(report,rows,cases,plan['launch_reference_ms'],plan['timing']['improvement_threshold'])
    if args.phase == 'matvec-development':
        report['comparison_to_greedy'] = report['decision']
        if report['complete']:
            report['decision'] = 'matvec-slice-complete'
        report['matvec_reference_ms'] = plan['launch_reference_ms']['4']
        blind = {agent:{'count':0,'completed':0} for agent in ('single','greedy')}
        for row in rows:
            if 'latency_ms' in row:
                row['at_or_below_reference'] = {k:row['latency_ms'] <= ms for k,ms in plan['launch_reference_ms']['4'].items()}
                if row['status']=='complete':
                    blind[row['agent']]['completed'] += 1
                    blind[row['agent']]['count'] += row['at_or_below_reference']['random_best_of_four']
            if row['agent']=='single':
                path = args.output/row['run']/'matched-evidence.json'
                row['matched_evidence'] = read_json(path) if path.is_file() else None
        report['at_or_below_random_four'] = blind
    if args.phase == 'matched-smoke':
        path = latest_attempt(args.output,job_name(args.phase,cases[0],'single',seeds[0]))/'matched-evidence.json'
        report['matched_evidence'] = read_json(path) if path.is_file() else None
        if report['complete']:
            identical = report['matched_evidence'] and report['matched_evidence']['identical']
            report['decision'] = 'paired-smoke-complete' if identical else 'missing-shared-evidence'
    report['attempts'] = attempts
    report['direct_pairs'] = {f'{case}-{seed}':data for (case,seed),data in measurements.items()}
    report['searches'] = rows
    report['matched_budget_ms'] = matched_budgets(rows,cases,seeds)
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
