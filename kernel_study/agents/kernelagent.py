"""Run the registered single-worker KernelAgent greedy search."""
import functools
import json
import math
import os
import re
import shutil
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from unittest.mock import patch

from ..budget import interruption
from ..judge.evaluate import run_worker
from ..judge.select import screen_candidate, select_final
from ..judge.policy import StaticRejection, check_source
from ..records import digest, environment, event, write_json
from ..tasks import kernelbench


def native_manager():
    from triton_kernel_agent.opt_manager import OptimizationManager
    return OptimizationManager


def worker_gpu():
    visible = (os.environ.get('CUDA_VISIBLE_DEVICES') or '0').split(',')[0].strip()
    return int(visible) if visible.isdecimal() else visible


def matvec_test_timeout(run_test):
    @functools.wraps(run_test)
    def verify(worker):
        if worker.worker_id == -1:
            with patch.object(worker, 'test_timeout_s', max(180, worker.test_timeout_s)):
                return run_test(worker)
        return run_test(worker)
    return verify


@contextmanager
def native_adapters(provider, seed, output, *, matvec=False):
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
            return None
        usage = provider.ledger.summary()
        index = usage['attempts']
        try:
            with (output/f'proposal_{index:03d}.py').open('x') as source:
                source.write(code or '')
        except FileExistsError:
            event(output/'candidates.jsonl','duplicate_extraction',attempt=index)
            return None
        event(output/'candidates.jsonl','extracted',attempt=index,round=round_number.get(),stage=stage.get(),
              source_hash=digest(code) if code else None,budget=usage)
        if code:
            try:
                check_source(code)
            except StaticRejection as exc:
                event(output/'candidates.jsonl','static_rejected',attempt=index,reason=str(exc))
                return None
        return code

    provider_config = ModelConfig(provider.model,[Transport])
    with ExitStack() as adapters:
        adapters.enter_context(patch.dict(models._get_model_name_to_config(),{provider.model:provider_config}))
        adapters.enter_context(patch.dict(models._provider_instances,{Transport:Transport()}))
        plan = kernelbench.experiment()
        specs = dict(plan['gpu_specs'], name=plan['gpu'])
        adapters.enter_context(patch.object(gpu_specs,'GPU_SPECS_DATABASE',
                                             {**gpu_specs.GPU_SPECS_DATABASE,specs['name']:specs}))
        adapters.enter_context(patch.dict(os.environ))
        os.environ.pop('OPENAI_API_KEY',None)
        if matvec:
            adapters.enter_context(patch.object(VerificationWorker, '_run_test',
                                                matvec_test_timeout(VerificationWorker._run_test)))
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
    root = kernelbench.upstream('KernelAgent')
    started = time.perf_counter()
    output.mkdir(parents=True,exist_ok=False)
    problem = output/'problem.py'
    problem.write_text(case['problem'] if 'problem' in case else kernelbench.task_path(case).read_text())
    starter = output/'baseline.py'
    shutil.copyfile(preflight/'input.py',starter)
    test = case['test'] if 'test' in case else kernelbench.test_source(case)
    (output/'test.py').write_text(test)
    opts = dict(config=str(root/'examples/configs/greedy.yaml'),num_workers=1,gpu_ids=[worker_gpu()],
                gpu_name=kernelbench.experiment()['gpu'],openai_model=provider.model,
                ncu_bin_path=shutil.which('ncu'),
                log_dir=output/'greedy',
                database_path=output/'greedy/program_db.json')
    if protocol.get('full_calibration'): opts['strategy_config'] = {'max_no_improvement':rounds}
    with native_adapters(provider, seed, output, matvec=(case['public_id']==4)):
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
        result = screen_candidate(source.read_text(),starter.read_text())
        if result is None:
            result = run_worker(req,output,f'candidate_{i:03}')
        if result['status']=='ok':
            eligible.append(dict(id=i, source=str(source)))
        history.append(dict(attempt=i,source_hash=digest(source.read_text()),result=result))
    start = dict(id=0, source=str(starter))
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
