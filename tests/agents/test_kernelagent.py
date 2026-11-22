import json

import pytest

from kernel_study.agents.kernelagent import native_adapters, native_manager, native_outcome
from kernel_study.budget import Ledger
from kernel_study.campaign.report import compare
from kernel_study.providers import MeteredChat
from kernel_study.records import event


def test_matvec_timeout_only_changes_the_initial_worker():
    from kernel_study.agents.kernelagent import matvec_test_timeout

    class Worker:
        test_timeout_s = 30

        def __init__(self, worker_id):
            self.worker_id = worker_id

        @matvec_test_timeout
        def run_test(self):
            return self.test_timeout_s

    initial, candidate = Worker(-1), Worker(0)
    assert initial.run_test() == 180
    assert candidate.run_test() == initial.test_timeout_s == 30


@pytest.mark.parametrize('visible,expected', [(None, 0), ('1,0', 1), ('GPU-abc', 'GPU-abc')])
def test_native_worker_uses_the_assigned_device(monkeypatch, visible, expected):
    from kernel_study.agents.kernelagent import worker_gpu

    if visible is None:
        monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    else:
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', visible)
    assert worker_gpu() == expected


def test_greedy_exhausted_without_a_winner_counts_as_fallback(tmp_path):
    for i in (1, 2):
        (tmp_path / f'proposal_{i:03d}.py').write_text('')
    final = {'candidate_id': 0}
    status, reason = native_outcome(tmp_path, {'attempts': 2}, {'success': False, 'total_rounds': 2}, final)
    assert status == 'complete'
    rows = [dict(case=4, seed=11, agent=agent, status=status, fallback=True, correct_new=False,
                 latency_ms=1.) for agent in ('single','greedy')]
    pair = {(4, 11): {'status':'ok', 'candidate':{'median_ms':1.}, 'incumbent':{'median_ms':1.}}}
    report = compare(rows, [4], [11], pair)
    assert report['paired_searches'] == 1 and report['fallbacks'] == {'single':1,'greedy':1}
    (tmp_path / 'proposal_002.py').unlink()
    assert native_outcome(tmp_path, {'attempts':2}, {'success':True}, final)[0] == 'invalid'


@pytest.mark.upstream
def test_pinned_upstream_proposal_and_repair_use_distinct_attempt_files(tmp_path,monkeypatch):
    import io
    import logging
    import urllib.request
    from types import SimpleNamespace

    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import OptimizationOrchestrator

    first = 'def kernel_function(x):\n    # Keep the input, then compute the elementwise result from the original value.\n    return x + 1'
    second = first.replace('x + 1','x + 2')
    replies = iter((first,second,'No candidate in this reply.'))
    def response(*args,**kwargs):
        reply = {'model':'test-model','choices':[{'message':{'content':next(replies)}}],
                 'usage':{'prompt_tokens':40,'completion_tokens':20}}
        return io.BytesIO(json.dumps(reply).encode())
    monkeypatch.setattr(urllib.request,'urlopen',response)
    ledger = Ledger(tmp_path,'callbacks',3,3,1.,1.,1.)
    provider = MeteredChat(ledger,tmp_path,model='test-model',endpoint='http://localhost:1')
    with native_adapters(provider,11,tmp_path):
        worker = VerificationWorker(0,tmp_path,tmp_path,openai_model='test-model')
        orchestrator = SimpleNamespace(logger=logging.getLogger('campaign-test'),artifact_dir=tmp_path,verification_worker=worker)
        generate = OptimizationOrchestrator._generate_optimized_kernel
        assert generate(orchestrator,'make a proposal',1)==first
        assert worker._refine_kernel(first,{'stderr':'test failure'},'pointwise','check output')==second
        assert generate(orchestrator,'try a different kernel',2) is None
        assert [(tmp_path/f'proposal_{i:03d}.py').read_text() for i in (1,2,3)] == [first,second,'']
        assert native_outcome(tmp_path, ledger.summary(), {'total_rounds':2}, {'candidate_id':0})[0]=='complete'
        worker._extract_code_from_response(first)
        assert native_outcome(tmp_path, ledger.summary(), {'total_rounds':2}, {'candidate_id':0})[0]=='invalid'
    calls=[e for e in map(json.loads,(tmp_path/'model_events.jsonl').read_text().splitlines()) if e['event']=='request']
    assert [e['stage'] for e in calls]==['proposal','repair','proposal']


def test_exhausted_diagnostic_call_budget_keeps_a_zero_proposal_search(tmp_path):
    event(tmp_path/'model_events.jsonl','budget_stop',reason='model_budget_exhausted')
    assert native_outcome(tmp_path,{'attempts':0,'calls':16},{'total_rounds':4},{'candidate_id':0})[0]=='complete'


@pytest.mark.upstream
def test_native_adapter_uses_upstream_parser_and_greedy_manager():
    from triton_kernel_agent.opt_manager import OptimizationManager
    assert native_manager() is OptimizationManager
