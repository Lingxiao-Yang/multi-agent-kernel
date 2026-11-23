"""Register and run the public comparison one search at a time."""
import fcntl
import json
import math
import os
import random
import subprocess
import sys

from ..agents.guidance import VERSION
from ..agents.kernelagent import benchmark_problem, run_native
from ..agents.local import search
from ..budget import Ledger
from ..judge.select import audit, measure_pair
from ..judge.evaluate import describe_ncu
from ..providers import MeteredChat
from ..records import ROOT, code_hashes, digest, read_json, write_json
from ..tasks import kernelbench, starts
from .preflight import check_device


def jobs(plan, phase):
    if phase=='matvec-development':
        return [(4,agent,seed) for seed in plan['development_seeds'] for agent in ('greedy','single')]
    if phase=='matched-smoke':
        spec = plan['phases'][phase]
        return [(spec['case'],agent,spec['seed']) for agent in ('greedy','single')]
    if phase in ('smoke','calibration'):
        phase_plan = plan['phases'][phase]
        agent = 'single' if phase=='smoke' else 'greedy'
        return [(phase_plan['case'],agent,phase_plan['seed'])]
    pairs = [(case,seed) for case in plan[phase] for seed in plan[phase+'_seeds']]
    rng = random.Random(20261121)
    rng.shuffle(pairs)
    result = []
    for case,seed in pairs:
        agents = list(plan['agents'])
        rng.shuffle(agents)
        result.extend((case,agent,seed) for agent in agents)
    return result


def phase_budget(plan, phase):
    return plan['phases']['development' if phase=='matvec-development' else phase]


def attempt_path(root, job, retry):
    base = root/job
    if retry:
        prior = base/'summary.json'
        if not prior.exists():
            raise ValueError('the original search has no completed interruption record')
        summary = read_json(prior)
        if (summary['status'],summary.get('stop_reason')) != ('interrupted','transport'):
            raise ValueError('only a transport interruption is eligible for one retry')
    path = root/(job+'-retry1') if retry else base
    if path.exists():
        raise ValueError(f'job directory already exists: {path}')
    return path


def latest_attempt(root, job):
    retry = root/(job+'-retry1')
    return retry if retry.exists() else root/job


def job_name(phase, case, agent, seed):
    return f'{phase}-{case}-{agent}-{seed}'


def shared_start_metrics(directory, source):
    from pathlib import Path
    sample = read_json(directory/'greedy/round001_ncu_metrics.json')
    if Path(sample['metadata']['kernel_file']).read_text() != source:
        raise ValueError('upstream NCU sample belongs to a different source')
    metrics = sample['metrics']
    (row,) = metrics.values()
    with (directory/'model_events.jsonl').open() as stream:
        request = next((e for e in map(json.loads, stream)
                        if e['event']=='request' and e['stage']=='analysis'), None)
    if not row or not request or not any(str(row) in msg['content'] for msg in request['messages']):
        raise ValueError('upstream analysis prompt does not contain this NCU sample')
    return metrics


def check_single_evidence(directory, evidence, specs):
    stages = {}
    with (directory/'model_events.jsonl').open() as stream:
        for request in map(json.loads, stream):
            if request['event'] != 'request':
                continue
            prompt = json.loads(next(m['content'] for m in request['messages'] if m['role']=='user'))
            if prompt['incumbent'].get('id',0) != 0:
                continue
            ncu = prompt['incumbent']['result'].get('profile',{}).get('ncu',{})
            identical = (ncu.get('metrics')==evidence['metrics'] and
                         ncu.get('roofline')==evidence['roofline'] and prompt['hardware']['specs']==specs)
            stage = request['stage']
            stages[stage] = stages.get(stage,True) and identical
    return dict(stages=stages, identical={'analyze','advise','implement'} <= stages.keys() and all(stages.values()),
                metric_count=sum(len(row) for row in evidence['metrics'].values()), metric_hash=digest(evidence['metrics']))


def register(directory, name, record):
    path = directory/(name+'.json')
    with (directory/(name+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            previous = read_json(path)
            if any(digest(previous.get(k))!=digest(v) for k,v in record.items() if k!='revision'):
                raise ValueError(f'{name} differs from the registered configuration')
        else:
            write_json(path, record)


def run_job(args, plan, case_id, agent, seed):
    job = job_name(args.phase,case_id,agent,seed)
    output = attempt_path(args.output,job,args.retry_transport)
    baseline = args.preflight/str(case_id)
    frozen = read_json(baseline/'preflight.json')
    stress = read_json(baseline/'stress.json')
    identity = ('native_source_hash','problem_hash')
    if any(stress[k]!=frozen[k] for k in identity) or not stress['checks'].get('native'):
        raise ValueError('public pressure checks belong to different starts')
    if 'native' not in frozen['checks']:
        raise ValueError('missing native preflight checks')
    register(args.output,f'case-{case_id}-preflight',dict(preflight=digest(frozen),stress=digest(stress)))
    case = dict(plan['cases'][str(case_id)], id=f'level1_{case_id}')
    case['problem'] = kernelbench.task_path(case).read_text()
    case['test'] = kernelbench.test_source(case)
    phase = phase_budget(plan,args.phase)
    cap, calls, rounds = phase['candidate_cap'],phase['call_cap'],phase['rounds']
    protocol = read_json(ROOT/'configs/protocol.json')
    protocol.update(plan['timing'],candidate_cap=cap, call_cap=calls, model=args.model,
                    endpoint=args.endpoint,input_usd_per_million=args.input_price,output_usd_per_million=args.output_price,
                    worker_timeout_s=600,search_space='native',feedback_mode='timing-and-ncu',
                    full_calibration=args.phase=='calibration',shared_api_guidance=VERSION,validation_seeds=plan['validation_seeds'],
                    final_validation_seeds=plan['final_validation_seeds'])
    ledger = Ledger(args.output,output.name,cap,calls,args.input_price,args.output_price,
                    plan['total_budget_usd']-plan['historical_cost_usd'])
    provider = MeteredChat(ledger,output,model=args.model,endpoint=args.endpoint,max_output_tokens=protocol['max_output_tokens'],
                           temperature=protocol['temperature'],input_price=args.input_price,output_price=args.output_price)
    if any(digest(source)!=frozen['native_source_hash'] for source in ((baseline/'input.py').read_text(),starts.start(case))):
        raise ValueError('starting code differs from preflight')
    if digest(case['problem']) != frozen['problem_hash']:
        raise ValueError('problem differs from preflight')
    case['problem'] = benchmark_problem(case['problem'],protocol['warmup_ms'])
    shared_start = args.phase in ('matched-smoke','matvec-development') and agent=='single'
    if shared_start:
        greedy = latest_attempt(args.output,job_name(args.phase,case_id,'greedy',seed))
        protocol['shared_ncu_metrics'] = shared_start_metrics(greedy,(baseline/'input.py').read_text())
    if agent=='single':
        result = search(case,'single',seed,protocol,provider,output)
    else:
        result = run_native(case,seed,protocol,provider,output,baseline,rounds)
    audit(output,agent)
    if shared_start and result['status']=='complete':
        report = check_single_evidence(output,describe_ncu(protocol['shared_ncu_metrics']),plan['gpu_specs'])
        write_json(output/'matched-evidence.json',report)
        if not report['identical']:
            raise ValueError('the actual model prompts did not match the registered hardware evidence')
    if args.phase in ('matched-smoke','matvec-development','development','holdout'):
        sides = [latest_attempt(args.output,job_name(args.phase,case_id,a,seed)) for a in ('single','greedy')]
        measured = measure_pair(args.output/'pairs'/f'{args.phase}-{case_id}-{seed}', *sides)
        if measured is not None and measured['status']!='ok':
            raise ValueError('direct finalist measurement failed; see the paired worker result')
    return result


def run_campaign(args, parser):
    plan = kernelbench.experiment()
    schedule = jobs(plan,args.phase)
    if not 0<=args.job<len(schedule): parser.error('job index outside phase')
    if not read_json(args.counters)['counter_access_verified']: parser.error('NCU counters unavailable')
    if not os.environ.get('OPENAI_API_KEY'): parser.error('OPENAI_API_KEY is missing')
    if not args.model or not all(p is not None and math.isfinite(p) and p>0 for p in (args.input_price,args.output_price)):
        parser.error('a pinned model snapshot and positive token prices are required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    args.preflight = args.preflight.resolve()
    device = check_device(plan)
    upstream = kernelbench.upstream('KernelAgent').resolve()
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(upstream),os.environ.get('PYTHONPATH','')])
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    lock = dict(model=args.model,endpoint=args.endpoint,input_price=args.input_price,output_price=args.output_price,
                gpu=plan['gpu'],driver=device['driver'],upstream=plan['upstream'],configuration=digest(plan))
    phase_lock = dict(revision=revision,code_hashes=code_hashes(),jobs=schedule)
    for name,record in [('experiment',lock),(args.phase+'-protocol',phase_lock)]:
        try:
            register(args.output,name,record)
        except ValueError as exc:
            parser.error(str(exc))
    case_id,agent,seed = schedule[args.job]
    if args.phase in ('matched-smoke','matvec-development','development','holdout'):
        pair = args.output/'pairs'/f'{args.phase}-{case_id}-{seed}'
        pair.mkdir(parents=True,exist_ok=True)
        try:
            register(pair,'device',dict(uuid=device['uuid'],node=device['node']))
        except ValueError as exc:
            parser.error(str(exc))
    if args.phase in ('matvec-development','development','holdout'):
        smoke = read_json(latest_attempt(args.output,job_name('smoke',*jobs(plan,'smoke')[0]))/'summary.json')
        if smoke['status']!='complete' or not any(h['result']['status'] in ('ok','incorrect','compile_error') for h in smoke['history']):
            parser.error('the native single smoke must reach GPU evaluation')
        calibration = read_json(latest_attempt(args.output,job_name('calibration',*jobs(plan,'calibration')[0]))/'summary.json')
        if calibration['status']!='complete' or calibration['upstream_rounds']!=plan['phases']['calibration']['rounds']:
            parser.error('matvec calibration incomplete')
    if args.phase=='holdout':
        freeze = read_json(args.output/'strategy_freeze.json')
        if freeze['code_hashes']!=code_hashes(): parser.error('code differs from development freeze')
    sys.path.insert(0,str(upstream))
    result = run_job(args,plan,case_id,agent,seed)
    print(result['status'])
    if result['status']!='complete': raise SystemExit(2)
