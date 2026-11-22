"""Run the frozen public comparison one search at a time."""
import fcntl
import functools
import json
import math
import os
import random
import re
import shutil
import sys
import subprocess
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from . import public_cases as public
from .api_guidance import VERSION
from .core import ROOT, digest, event, read_json, write_json
from .reference_search import Ledger, MeteredChat, interruption
from .public_referee import audit, measure_pair, select_final
from .search import run_worker, search


def jobs(plan, phase):
    if phase in ('smoke','calibration'):
        phase_plan = plan['phases'][phase]
        return [(phase_plan['case'],'greedy',phase_plan['seed'])]
    pairs = [(case,seed) for case in plan[phase] for seed in plan[phase+'_seeds']]
    rng = random.Random(20261121)
    rng.shuffle(pairs)
    result = []
    for case,seed in pairs:
        agents = list(plan['agents'])
        rng.shuffle(agents)
        result.extend((case,agent,seed) for agent in agents)
    return result


def native_manager():
    from triton_kernel_agent.opt_manager import OptimizationManager
    return OptimizationManager


@contextmanager
def native_adapters(provider, seed, output):
    from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import OptimizationOrchestrator
    from kernel_perf_agent.kernel_opt.diagnose_prompt import gpu_specs
    from triton_kernel_agent.worker import VerificationWorker
    from utils.providers import models
    from utils.providers.base import LLMResponse
    from utils.providers.model_config import ModelConfig

    stage = ContextVar('stage',default='analysis')
    round_number = ContextVar('round_number',default=None)
    class Transport:
        name = 'openai'
        def is_available(self): return True
        def get_response(self, model_name, messages, **kwargs):
            if model_name != provider.model:
                raise ValueError('unexpected model selection')
            reply = provider.send_messages(messages,stage.get(),seed+provider.ledger.summary()['calls']+1,
                                           max_tokens=kwargs.get('max_tokens'))
            return LLMResponse(content=reply.text,model=reply.model,provider=self.name,usage=reply.metadata['usage'])

    def tagged(fn, label):
        @functools.wraps(fn)
        def call(self,*args,**kwargs):
            token = stage.set(label)
            run = round_number.set((args[1] if len(args)>1 else kwargs['round_num']) if label=='proposal' else round_number.get())
            try: return fn(self,*args,**kwargs)
            finally:
                round_number.reset(run)
                stage.reset(token)
        return call

    extract = VerificationWorker._extract_code_from_response
    def save_source(worker,*args,**kwargs):
        code = extract(worker,*args,**kwargs)
        if stage.get() not in ('proposal','repair'):
            event(output/'candidates.jsonl','unattributed_extraction')
            return code
        usage = provider.ledger.summary()
        index = usage['attempts']
        try:
            with (output/f'proposal_{index:03d}.py').open('x') as source:
                source.write(code or '')
        except FileExistsError:
            event(output/'candidates.jsonl','duplicate_extraction',attempt=index)
            return code
        event(output/'candidates.jsonl','extracted',attempt=index,round=round_number.get(),stage=stage.get(),
              source_hash=digest(code) if code else None,budget=usage)
        return code

    provider_config = ModelConfig(provider.model,[Transport])
    with ExitStack() as adapters:
        adapters.enter_context(patch.dict(models._get_model_name_to_config(),{provider.model:provider_config}))
        adapters.enter_context(patch.dict(models._provider_instances,{Transport:Transport()}))
        plan = public.experiment()
        specs = dict(plan['gpu_specs'], name=plan['gpu'])
        adapters.enter_context(patch.object(gpu_specs,'GPU_SPECS_DATABASE',
                                             {**gpu_specs.GPU_SPECS_DATABASE,specs['name']:specs}))
        adapters.enter_context(patch.dict(os.environ))
        os.environ.pop('OPENAI_API_KEY',None)
        for owner,method,label in [(OptimizationOrchestrator,'_generate_optimized_kernel','proposal'),
                                   (VerificationWorker,'_refine_kernel','repair')]:
            adapters.enter_context(patch.object(owner,method,tagged(getattr(owner,method),label)))
        adapters.enter_context(patch.object(VerificationWorker,'_extract_code_from_response',save_source))
        yield


def native_outcome(output, usage, native, final):
    reason = interruption(output)
    if reason:
        return 'interrupted', reason
    expected = set(range(1, usage['attempts']+1))
    recorded = {int(p.stem.split('_')[1]) for p in output.glob('proposal_*.py')}
    log = output/'candidates.jsonl'
    mislabelled = log.exists() and any(json.loads(line)['event'] in ('duplicate_extraction','unattributed_extraction')
                                      for line in log.read_text().splitlines())
    if recorded!=expected or mislabelled:
        return 'invalid', 'candidate_log_mismatch'
    if not final:
        return 'final_failed', 'no_valid_finalist'
    events = output/'model_events.jsonl'
    exhausted = events.exists() and any(e['event']=='budget_stop' and e['reason'] in
        ('candidate_cap','call_cap','model_budget_exhausted','token_safety_cap')
        for e in map(json.loads,events.read_text().splitlines()))
    if (not expected and not exhausted) or native.get('error'):
        return 'interrupted', 'upstream_stopped'
    return 'complete', 'search_finished'


def run_native(case, seed, protocol, provider, output, preflight, rounds):
    from .core import environment
    root = public.upstream('KernelAgent')
    started = time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    problem = output/'problem.py'
    shutil.copyfile(public.task_path(case),problem)
    starter = output/'baseline.py'
    shutil.copyfile(preflight/'input.py',starter)
    if case['public_id']==4:
        test = (root/'examples/optimize_01_matvec/test.py').read_text()
    else:
        test = ('import torch\nfrom kernel import kernel_function\nfrom kernel_study import public_cases as p\n'
                'def test_kernel():\n    with torch.no_grad():\n'
                f"        case = p.experiment()['cases'][{str(case['public_id'])!r}]\n"
                "        model, inputs, buffers = p.task_inputs(case,101)\n"
                "        p.check_forward(kernel_function,buffers,model(*inputs),case)\n"
                "    return True\nif __name__ == '__main__':\n    test_kernel()\n    print('PASS')\n")
    (output/'test.py').write_text(test)
    opts = dict(config=str(root/'examples/configs/greedy.yaml'),num_workers=1,gpu_ids=[0],
                gpu_name=public.experiment()['gpu'],openai_model=provider.model,
                ncu_bin_path=shutil.which('ncu'),
                log_dir=output/'greedy',
                database_path=output/'greedy/program_db.json')
    if protocol.get('full_calibration'): opts['strategy_config'] = {'max_no_improvement':rounds}
    with native_adapters(provider, seed, output):
        manager = native_manager()(**opts)
        native = manager.run_optimization(initial_kernel=starter.read_text(),problem_file=problem,
                                          test_code=test,max_rounds=rounds)
    usage = provider.ledger.summary()
    starts = native.get('initial_kernel_time_ms')
    best = starts if starts is not None and math.isfinite(starts) else None
    winners = {int(r):float(t) for r,t in re.findall(r'Round (\d+) best: worker \d+ at ([\d.]+) ms',
                                                     (output/'greedy/manager.log').read_text())}
    curve = []
    for i in range(1,native.get('total_rounds',0)+1):
        if i in winners: best = min(best,winners[i]) if best is not None else winners[i]
        curve.append({'round':i,'round_best_ms':winners.get(i),'best_ms':best})
    selected = native.get('kernel_code')
    candidates = sorted(output.glob('proposal_*.py'))
    history, eligible = [], []
    for source in candidates:
        i = int(source.stem.split('_')[1])
        req = dict(case=case,source=str(source),interface='native',incumbent_source=str(starter),protocol=protocol)
        result = run_worker(req,output,f'candidate_{i:03}') if source.stat().st_size else {'status':'no_submission'}
        if result['status']=='ok':
            eligible.append(dict(id=i, source=str(source), config=None))
        history.append(dict(attempt=i,source_hash=digest(source.read_text()),result=result))
    start = dict(id=0, source=str(starter), config=None)
    final, selection = select_final(output, case, protocol, 'native', start, eligible)
    native_choices = [starter, *candidates]
    proposed = next((p for p in native_choices if selected and p.read_text()==selected),None)
    native_time = native.get('best_time_ms')
    upstream_selected = dict(source_hash=digest(selected) if selected else None,
                             source=proposed.name if proposed else None,
                             latency_ms=native_time if native_time is not None and math.isfinite(native_time) else None)
    summary = dict(manifest=dict(case=case,seed=seed,variant='greedy',protocol=protocol,environment=environment()),
                   final=final,selection=selection,upstream_selected=upstream_selected,history=history,budget=usage,upstream_rounds=native.get('total_rounds'),rounds=curve,
                   upstream_success=native.get('success',False),wall_seconds=time.perf_counter()-started,
                   estimated_inference_usd=usage['completed_usd'],budget_committed_usd=usage['committed_usd'])
    summary['status'], summary['stop_reason'] = native_outcome(output, usage, native, final)
    write_json(output/'summary.json',summary)
    return summary


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
    hold = read_json(baseline/'hold.json')
    stress = read_json(baseline/'stress.json')
    identity = ('device_source_hash','native_source_hash','problem_hash','launch')
    if any(record[k]!=frozen[k] for record in (stress,hold) for k in identity) or set(stress['checks'])!={'native','declarative'}:
        raise ValueError('public pressure or timing checks belong to different starts')
    if not frozen['identical_outputs'] or not hold['interface_agreement'] or hold['self']['confirmations']>1:
        raise ValueError('public preflight acceptance incomplete')
    case = dict(plan['cases'][str(case_id)],id=f'level1_{case_id}',problem=public.task_path(plan['cases'][str(case_id)]).read_text())
    _, dims, _ = public.layout(case)
    case['shape'] = [dims['M'],dims['N']] + ([dims['K']] if 'K' in dims else [])
    phase = plan['phases'][args.phase]
    cap, calls, rounds = phase['candidate_cap'],phase['call_cap'],phase['rounds']
    protocol = read_json(ROOT/'configs/protocol.json')
    protocol.update(plan['timing'],candidate_cap=cap, call_cap=calls, model=args.model,
                    endpoint=args.endpoint,input_usd_per_million=args.input_price,output_usd_per_million=args.output_price,
                    improvement_threshold=hold['self']['threshold'],worker_timeout_s=600,search_space='source-v1',
                    full_calibration=args.phase=='calibration',shared_api_guidance=VERSION,validation_seeds=plan['validation_seeds'],
                    final_validation_seeds=plan['final_validation_seeds'])
    ledger = Ledger(args.output,output.name,cap,calls,args.input_price,args.output_price,
                    plan['total_budget_usd']-plan['historical_cost_usd'])
    provider = MeteredChat(ledger,output,model=args.model,endpoint=args.endpoint,max_output_tokens=protocol['max_output_tokens'],
                           temperature=protocol['temperature'],input_price=args.input_price,output_price=args.output_price)
    if digest((baseline/'input.py').read_text()) != frozen['native_source_hash'] or digest(public.start(case)[0]) != frozen['device_source_hash']:
        raise ValueError('starting code differs from preflight')
    if digest(case['problem']) != frozen['problem_hash'] or public.start(case)[1] != frozen['launch']:
        raise ValueError('problem or launch differs from preflight')
    if agent=='single':
        result = search(case,'single',seed,protocol,provider,output)
    else:
        result = run_native(case,seed,protocol,provider,output,baseline,rounds)
    audit(output,agent)
    if args.phase in ('development','holdout'):
        sides = [latest_attempt(args.output,job_name(args.phase,case_id,a,seed)) for a in ('single','greedy')]
        measured = measure_pair(args.output/'pairs'/f'{args.phase}-{case_id}-{seed}', *sides)
        if measured is not None and measured['status']!='ok':
            raise ValueError('direct finalist measurement failed; see the paired worker result')
    return result


def run_campaign(args, parser):
    plan = public.experiment()
    schedule = jobs(plan,args.phase)
    if not 0<=args.job<len(schedule): parser.error('job index outside phase')
    if not read_json(args.counters)['counter_access_verified']: parser.error('NCU counters unavailable')
    if not os.environ.get('OPENAI_API_KEY'): parser.error('OPENAI_API_KEY is missing')
    if not args.model or not all(p is not None and math.isfinite(p) and p>0 for p in (args.input_price,args.output_price)):
        parser.error('a pinned model snapshot and positive token prices are required')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    args.preflight = args.preflight.resolve()
    from .public_preflight import check_device
    device = check_device(plan)
    upstream = public.upstream('KernelAgent').resolve()
    os.environ['PYTHONPATH'] = os.pathsep.join([str(ROOT),str(upstream),os.environ.get('PYTHONPATH','')])
    revision = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    lock = dict(model=args.model,endpoint=args.endpoint,input_price=args.input_price,output_price=args.output_price,
                gpu=device['gpu'],device_uuid_sha256=plan['device_uuid_sha256'],upstream=plan['upstream'],
                configuration=digest(plan),
                preflight={str(c):{stage:digest(read_json(args.preflight/str(c)/(stage+'.json')))
                                   for stage in ('preflight','stress','hold')}
                           for c in plan['development']+plan['holdout']})
    phase_lock = dict(revision=revision,code_hashes=code_hashes(),jobs=schedule)
    for name,record in [('experiment',lock),(args.phase+'-protocol',phase_lock)]:
        try:
            register(args.output,name,record)
        except ValueError as exc:
            parser.error(str(exc))
    if args.phase in ('calibration','development','holdout'):
        smoke = read_json(latest_attempt(args.output,job_name('smoke',*jobs(plan,'smoke')[0]))/'summary.json')
        if smoke['status']!='complete' or not any(h['result']['status']=='ok' for h in smoke['history']):
            parser.error('a correct real smoke proposal is required')
    if args.phase in ('development','holdout'):
        calibration = read_json(latest_attempt(args.output,job_name('calibration',*jobs(plan,'calibration')[0]))/'summary.json')
        if calibration['status']!='complete' or calibration['upstream_rounds']!=plan['phases']['calibration']['rounds']:
            parser.error('matvec calibration incomplete')
    if args.phase=='holdout':
        freeze = read_json(args.output/'strategy_freeze.json')
        if freeze['code_hashes']!=code_hashes(): parser.error('code differs from development freeze')
    sys.path.insert(0,str(upstream))
    result = run_job(args,plan,*schedule[args.job])
    print(result['status'])
    if result['status']!='complete': raise SystemExit(2)


def code_hashes():
    modules = ('agent_campaign','api_guidance','bench','core','knowledge','providers',
               'public_cases','public_kernels','public_preflight','public_referee','reference_search','replay',
               'search','source_policy','source_space','worker')
    hashes = {name+'.py':digest((ROOT/'kernel_study'/(name+'.py')).read_text()) for name in modules}
    for path in ('configs/agent_baseline.json','configs/protocol.json','knowledge/cards.json'):
        hashes[path] = digest((ROOT/path).read_text())
    return hashes
