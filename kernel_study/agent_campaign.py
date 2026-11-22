"""Run the frozen public comparison one search at a time."""
import argparse
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
from contextlib import ExitStack
from contextvars import ContextVar
from pathlib import Path
from unittest.mock import patch

from . import public_cases as public
from .api_guidance import VERSION
from .core import ROOT, digest, event, read_json, write_json
from .reference_search import Ledger, MeteredChat
from .search import run_worker, search


def jobs(plan, phase):
    if phase == 'smoke': return [(4,'greedy',771)]
    if phase == 'calibration': return [(4,'greedy',772)]
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


def a800_specs():
    return dict(name=public.experiment()['gpu'],architecture='Ampere',peak_memory_bw_gbps=2039,
                peak_fp32_tflops=19.5,peak_fp16_tflops=312,peak_bf16_tflops=312,
                sm_count=108,max_threads_per_sm=2048,l1_cache_kb=192,l2_cache_mb=40,
                memory_gb=80,memory_type='HBM2e')


def run_native(case, seed, protocol, provider, output, preflight, rounds):
    from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import OptimizationOrchestrator
    from kernel_perf_agent.kernel_opt.diagnose_prompt import gpu_specs
    from triton_kernel_agent.worker import VerificationWorker
    from utils.providers import models
    from utils.providers.base import LLMResponse
    from utils.providers.model_config import ModelConfig
    from .core import environment

    root = public.upstream('KernelAgent')
    started = time.perf_counter()
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
            run = round_number.set(args[1] if label=='proposal' else round_number.get())
            try: return fn(self,*args,**kwargs)
            finally:
                round_number.reset(run)
                stage.reset(token)
        return call

    extract = VerificationWorker._extract_code_from_response
    def save_source(worker,*args,**kwargs):
        code = extract(worker,*args,**kwargs)
        if stage.get() in ('proposal','repair'):
            usage = provider.ledger.summary()
            index = usage['attempts']
            if code: (output/f'proposal_{index:03d}.py').write_text(code)
            event(output/'candidates.jsonl','extracted',attempt=index,round=round_number.get(),stage=stage.get(),
                  source_hash=digest(code) if code else None,budget=usage)
        return code

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
    if rounds==8: opts['strategy_config'] = {'max_no_improvement':8}
    provider_config = ModelConfig(provider.model,[Transport])
    with ExitStack() as adapters:
        adapters.enter_context(patch.dict(models._get_model_name_to_config(),{provider.model:provider_config}))
        adapters.enter_context(patch.dict(models._provider_instances,{Transport:Transport()}))
        specs = a800_specs()
        adapters.enter_context(patch.object(gpu_specs,'GPU_SPECS_DATABASE',
                                             {**gpu_specs.GPU_SPECS_DATABASE,specs['name']:specs}))
        adapters.enter_context(patch.dict(os.environ))
        os.environ.pop('OPENAI_API_KEY',None)
        for owner,method,label in [(OptimizationOrchestrator,'_generate_optimized_kernel','proposal'),
                                   (VerificationWorker,'_refine_kernel','repair')]:
            adapters.enter_context(patch.object(owner,method,tagged(getattr(owner,method),label)))
        adapters.enter_context(patch.object(VerificationWorker,'_extract_code_from_response',save_source))
        manager = native_manager()(**opts)
        native = manager.run_optimization(initial_kernel=starter.read_text(),problem_file=problem,
                                          test_code=test,max_rounds=rounds)
    usage = provider.ledger.summary()
    events_file = output/'model_events.jsonl'
    events = [json.loads(line) for line in events_file.read_text().splitlines()] if events_file.exists() else []
    stops = [e for e in events if e['event']=='request_error' or
             (e['event']=='budget_stop' and e['reason']=='dollar_cap_reservation')]
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
    history = []
    for source in candidates:
        i = int(source.stem.split('_')[1])
        req = dict(case=case,source=str(source),interface='native',incumbent_source=str(starter),protocol=protocol)
        result = run_worker(req,output,f'candidate_{i:03}')
        history.append(dict(attempt=i,source_hash=digest(source.read_text()),result=result))
    from .reference_report import changed_program
    proposed = next((p for p in candidates if p.read_text()==selected and changed_program(selected,starter.read_text())),None)
    final = None
    for source in ([proposed,starter] if proposed else [starter]):
        i = 0 if source==starter else int(source.stem.split('_')[1])
        req = dict(case=case,source=str(source),interface='native',incumbent_source=str(starter),
                   protocol=protocol,final=True,library=True)
        result = run_worker(req,output,f'final_{i:03}')
        if result['status']=='ok':
            final = dict(candidate_id=i,result=result)
            break
    summary = dict(manifest=dict(case=case,seed=seed,variant='greedy',protocol=protocol,environment=environment()),
                   final=final,history=history,budget=usage,upstream_rounds=native.get('total_rounds'),rounds=curve,
                   upstream_success=native.get('success',False),wall_seconds=time.perf_counter()-started,
                   estimated_inference_usd=usage['completed_usd'],budget_committed_usd=usage['committed_usd'])
    summary['status'] = 'complete' if final and native.get('success') and usage['attempts'] and not stops else 'interrupted'
    summary['stop_reason'] = stops[-1].get('reason',stops[-1].get('error')) if stops else summary['status']
    write_json(output/'summary.json',summary)
    return summary


def run_job(args, plan, case_id, agent, seed):
    job = f'{args.phase}-{case_id}-{agent}-{seed}'
    output = args.output/job
    if output.exists():
        raise ValueError(f'job directory already exists: {output}')
    baseline = args.preflight/str(case_id)
    frozen = read_json(baseline/'preflight.json')
    hold = read_json(baseline/'hold.json')
    if not frozen['identical_outputs'] or not hold['interface_agreement'] or hold['self']['confirmations']>1:
        raise ValueError('public preflight acceptance incomplete')
    case = dict(plan['cases'][str(case_id)],id=f'level1_{case_id}',problem=public.task_path(plan['cases'][str(case_id)]).read_text())
    _, dims, _ = public.layout(case)
    case['shape'] = [dims['M'],dims['N']] + ([dims['K']] if 'K' in dims else [])
    cap, calls, rounds = plan['candidate_cap'],plan['call_cap'],4
    if args.phase=='smoke': cap,calls,rounds = 1,16,1
    if args.phase=='calibration': cap,calls,rounds = 32,128,8
    protocol = read_json(ROOT/'configs/protocol.json')
    protocol.update(plan['timing'],candidate_cap=cap, call_cap=calls, model=args.model,
                    endpoint=args.endpoint,input_usd_per_million=args.input_price,output_usd_per_million=args.output_price,
                    improvement_threshold=hold['self']['threshold'],worker_timeout_s=600,search_space='source-v1',
                    shared_api_guidance=VERSION,validation_seeds=plan['validation_seeds'],
                    final_validation_seeds=plan['final_validation_seeds'])
    ledger = Ledger(args.output,job,cap,calls,args.input_price,args.output_price,
                    plan['total_budget_usd']-plan['historical_cost_usd'])
    provider = MeteredChat(ledger,output,model=args.model,endpoint=args.endpoint,max_output_tokens=4096,
                           temperature=protocol['temperature'],input_price=args.input_price,output_price=args.output_price)
    if digest((baseline/'input.py').read_text()) != frozen['native_source_hash'] or digest(public.start(case)[0]) != frozen['device_source_hash']:
        raise ValueError('starting code differs from preflight')
    if digest(case['problem']) != frozen['problem_hash'] or public.start(case)[1] != frozen['launch']:
        raise ValueError('problem or launch differs from preflight')
    if agent=='single':
        result = search(case,'single',seed,protocol,provider,output)
    else:
        result = run_native(case,seed,protocol,provider,output,baseline,rounds)
    if (output/'summary.json').exists() and 'final' in result:
        from .reference_report import audit
        audit(output,agent)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['smoke','calibration','development','holdout'])
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--counters',type=Path,required=True)
    parser.add_argument('--job',type=int,required=True,help='Zero-based index in the phase')
    parser.add_argument('--model',default=os.environ.get('KERNEL_BASELINE_MODEL'))
    parser.add_argument('--endpoint',default=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'))
    parser.add_argument('--input-price',type=float,default=os.environ.get('KERNEL_BASELINE_INPUT_PRICE'))
    parser.add_argument('--output-price',type=float,default=os.environ.get('KERNEL_BASELINE_OUTPUT_PRICE'))
    args = parser.parse_args()
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
                preflight={str(c):digest(read_json(args.preflight/str(c)/'preflight.json')) for c in plan['development']+plan['holdout']})
    phase_lock = dict(revision=revision,code_hashes=code_hashes(),jobs=schedule)
    for name,record in [('experiment',lock),(args.phase+'-protocol',phase_lock)]:
        lockfile = args.output/(name+'.json')
        if lockfile.exists():
            if digest(read_json(lockfile))!=digest(record): parser.error(f'{name} differs from the registered configuration')
        else:
            write_json(lockfile,record)
    if args.phase in ('calibration','development','holdout'):
        smoke = read_json(args.output/'smoke-4-greedy-771/summary.json')
        if smoke['status']!='complete' or not any(h['result']['status']=='ok' for h in smoke['history']):
            parser.error('a correct real smoke proposal is required')
    if args.phase in ('development','holdout'):
        calibration = read_json(args.output/'calibration-4-greedy-772/summary.json')
        if calibration['status']!='complete' or calibration['upstream_rounds']!=8:
            parser.error('matvec calibration incomplete')
    if args.phase=='holdout':
        freeze = read_json(args.output/'strategy_freeze.json')
        if freeze['code_hashes']!=code_hashes(): parser.error('code differs from development freeze')
    sys.path.insert(0,str(upstream))
    result = run_job(args,plan,*schedule[args.job])
    print(result['status'])
    if result['status']!='complete': raise SystemExit(2)


def code_hashes():
    return {p.name:digest(p.read_text()) for p in sorted((ROOT/'kernel_study').glob('*.py'))}


if __name__ == '__main__':
    main()
