import json

import pytest

from kernel_study.core import read_json, write_json


def test_greedy_exhausted_without_a_winner_counts_as_fallback(tmp_path):
    from kernel_study.agent_campaign import native_outcome
    from kernel_study.reference_report import compare
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


@pytest.mark.parametrize('interface', ['native','declarative'])
def test_final_selection_checks_every_candidate_against_the_current_winner(tmp_path, monkeypatch, interface):
    from kernel_study import public_referee as referee
    start = {'id':0, 'source':str(tmp_path/'baseline.py'), 'config':None}
    candidates = []
    for i in range(5):
        path = tmp_path / ('baseline.py' if i == 0 else f'program_{i}.py')
        path.write_text(f'def kernel(x):\n    return x + {i}\n')
        if i: candidates.append(dict(id=i, source=str(path), config=None))
    trials = []
    def worker(request, directory, label):
        i = next(item['id'] for item in [start,*candidates] if item['source'] == request['source'])
        trials.append((i, request['incumbent_source'], request.get('final'), request['interface']))
        if i == 3: return {'status':'incorrect'}
        return {'status':'ok', 'confirmed':i in (2,3), 'candidate':{'median_ms':float(i or 5)}}
    monkeypatch.setattr(referee, 'run_worker', worker)
    final, checks = referee.select_final(tmp_path, {}, {}, interface, start, candidates)
    assert final['candidate_id'] == 2
    assert [c['attempt'] for c in checks] == [1,2,3,4]
    assert [c['selected'] for c in checks] == [False,True,False,False]
    assert all(t[2] and t[3] == interface for t in trials)
    assert trials[2][1] == candidates[1]['source']
    assert trials[3][1] == candidates[1]['source']


def test_primary_ratio_uses_only_the_direct_pair():
    from kernel_study.reference_report import compare
    rows = [dict(case=40, seed=11, agent=a, correct_new=True, fallback=False, status='complete',
                 latency_ms=t) for a,t in [('single',9.), ('greedy',1.)]]
    direct = {(40,11): {'status':'ok', 'candidate':{'median_ms':1.04}, 'incumbent':{'median_ms':1.}}}
    assert compare(rows, [40], [11], direct)['ratio'] == pytest.approx(1.04)
    assert not compare(rows, [40], [11], {})['complete']


def test_pair_worker_receives_both_recorded_finalists_and_reuses_raw_samples(tmp_path, monkeypatch):
    from kernel_study import public_referee as referee
    directories = {}
    for agent, interface in [('single','declarative'),('greedy','native')]:
        directory = tmp_path/agent
        directory.mkdir()
        (directory/'winner.py').write_text('pass')
        write_json(directory/'summary.json', {'manifest':{'case':{'public_id':4},'protocol':{'p':1},'seed':11}})
        write_json(directory/'audit.json', dict(status='complete', finalist=dict(source='winner.py',config=None,interface=interface)))
        directories[agent] = directory
    calls = []
    def worker(req, where, label):
        calls.append(req)
        result = dict(status='ok',candidate={'median_ms':1.,'samples_ms':[1.]},incumbent={'median_ms':2.,'samples_ms':[2.]})
        write_json(where/(label+'.result.json'), result)
        return result
    monkeypatch.setattr(referee,'run_worker',worker)
    for _ in range(2):
        result = referee.measure_pair(tmp_path/'pair', directories['single'], directories['greedy'])
        assert result['incumbent']['samples_ms'] == [2.]
    assert len(calls) == 1
    req = calls[0]
    assert req['source'] == str(directories['single']/'winner.py')
    assert req['incumbent_source'] == str(directories['greedy']/'winner.py')
    assert (req['interface'],req['incumbent_interface']) == ('declarative','native')


def test_only_one_new_attempt_after_transport_failure(tmp_path):
    from kernel_study.agent_campaign import attempt_path
    job = 'development-4-greedy-11'
    base = attempt_path(tmp_path, job, False)
    base.mkdir()
    for reason in ('dollar_cap_reservation','provider_error','candidate_log_mismatch'):
        write_json(base/'summary.json', dict(status='interrupted',stop_reason=reason))
        with pytest.raises(ValueError): attempt_path(tmp_path,job,True)
    write_json(base/'summary.json',dict(status='interrupted',stop_reason='transport'))
    second = attempt_path(tmp_path,job,True)
    assert second != base
    second.mkdir()
    with pytest.raises(ValueError): attempt_path(tmp_path,job,True)
    assert read_json(base/'summary.json')['stop_reason']=='transport'


def test_transport_failure_is_logged_and_upstream_cannot_silently_call_again(tmp_path,monkeypatch):
    import urllib.request
    import urllib.error
    from kernel_study.reference_search import Ledger,MeteredChat
    def offline(*args,**kwargs): raise urllib.error.URLError('unreachable')
    monkeypatch.setattr(urllib.request,'urlopen',offline)
    ledger = Ledger(tmp_path, 'sample', 4, 16, 1., 1., 1.)
    chat = MeteredChat(ledger,tmp_path,model='pinned',endpoint='http://localhost:1')
    with pytest.raises(Exception): chat.send_messages([], 'proposal', 1)
    with pytest.raises(Exception): chat.send_messages([], 'repair', 2)
    assert ledger.summary()['calls']==1 and ledger.summary()['committed_usd']>0
    failures = [e for e in map(json.loads,(tmp_path/'model_events.jsonl').read_text().splitlines()) if e['event']=='request_error']
    assert [e['reason'] for e in failures]==['transport']


def test_frozen_code_and_phase_settings():
    from kernel_study.agent_campaign import code_hashes
    from kernel_study.public_cases import experiment
    plan=experiment()
    frozen=code_hashes()
    assert 'reference_report.py' not in frozen and 'public_referee.py' in frozen
    assert 'public_cases.py' in frozen and 'agent_campaign.py' in frozen
    assert plan['gpu_specs']['peak_memory_bw_gbps'] == 2039
    assert [(v['candidate_cap'],v['call_cap'],v['rounds']) for v in
            (plan['phases'][k] for k in ('smoke','calibration','development','holdout'))] == [
                (1,16,1),(32,128,8),(4,16,4),(4,16,4)]


def test_stress_uses_the_fp64_error_of_eager_and_rejects_nonfinite():
    torch=pytest.importorskip('torch')
    from kernel_study.public_cases import check_forward
    x = torch.tensor([[1., 1.]])
    expected = torch.tensor([[.9, 1.1]])
    reference = torch.ones_like(expected, dtype=torch.float64)
    case = dict(atol=1.e-6, rtol=0.)
    def check(output):
        return check_forward(lambda *_: output,[x],expected,case,
                             reference_chunks=[(slice(None),reference)])
    assert check(torch.tensor([[1.1, .9]]))['elements']==2
    for invalid in (torch.tensor([[1.2,1.]]), torch.tensor([[float('nan'),1.]])):
        with pytest.raises(AssertionError): check(invalid)


def test_declarative_validation_poisons_each_fresh_output(tmp_path):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    from kernel_study import public_cases, source_space
    case={**public_cases.experiment()['cases']['26'],'output_shape':[2,17]}
    source,config=public_cases.start(case)
    source=source.replace('i<TOTAL)', 'i<TOTAL-1)')
    path=tmp_path/'partial.py'
    path.write_text(source)
    x=torch.ones((2,17),device='cuda')
    forward,_=source_space.prepare(case,config,[x],path,fresh_output=True,poison_output=True)
    for _ in range(2):
        assert torch.isnan(forward().flatten()[-1])


def test_public_starts_read_their_own_device_modules(monkeypatch):
    from kernel_study import public_cases as p
    original=p.device_function
    def read(path,name):
        assert path.name != 'kernels.py'
        return original(path,name)
    monkeypatch.setattr(p,'device_function',read)
    for case in ('2','23','40'):
        assert p.start(p.experiment()['cases'][case])[1]['launches']


def test_pinned_upstream_proposal_and_repair_use_distinct_attempt_files(tmp_path,monkeypatch):
    import io
    import logging
    import urllib.request
    from types import SimpleNamespace
    pytest.importorskip('triton_kernel_agent')
    from kernel_study.agent_campaign import native_adapters, native_outcome
    from kernel_study.reference_search import Ledger, MeteredChat
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
    from kernel_study.agent_campaign import native_outcome
    from kernel_study.core import event
    event(tmp_path/'model_events.jsonl','budget_stop',reason='model_budget_exhausted')
    assert native_outcome(tmp_path,{'attempts':0,'calls':16},{'total_rounds':4},{'candidate_id':0})[0]=='complete'


def test_retry_report_charges_both_runs_but_counts_one_search(tmp_path):
    from kernel_study.reference_report import phase_rows
    name = 'development-4-single-11'
    for suffix, status, charge in [('', 'interrupted',.12),('-retry1','complete',.05)]:
        row=dict(agent='single',case=4,seed=11,status=status,stop_reason='transport' if not suffix else 'candidate_cap',
                 budget={'attempts':1},cost_usd=0. if not suffix else charge,committed_usd=charge,
                 candidates=[{'attempt':1,'usd':charge}])
        write_json(tmp_path/(name+suffix)/'audit.json',row)
    rows,attempts,_=phase_rows(tmp_path,'development',[4],[11])
    assert len(rows)==1 and len(attempts)==2
    assert rows[0]['total_committed_usd']==pytest.approx(.17)
    assert rows[0]['total_completed_usd']==pytest.approx(.05)
    assert rows[0]['total_attempts']==2
    assert rows[0]['matched_candidates'][0]['attempt']==2


def test_reporting_only_revision_preserves_the_registered_search(tmp_path):
    from kernel_study.agent_campaign import register
    registered=dict(revision='before',code_hashes={'worker.py':'abc'},jobs=[[4,'greedy',11]])
    register(tmp_path,'development-protocol',registered)
    register(tmp_path,'development-protocol',dict(registered,revision='report-only'))
    assert read_json(tmp_path/'development-protocol.json')==registered
    with pytest.raises(ValueError):
        register(tmp_path,'development-protocol',dict(registered,revision='worker-changed',code_hashes={'worker.py':'def'}))


def test_local_search_cap_takes_precedence_over_an_unsent_charge(tmp_path):
    from kernel_study.reference_search import Ledger
    ledger=Ledger(tmp_path,'capped',1,1,1.,1.,.002)
    call=ledger.reserve('proposal',1000,1000)
    ledger.settle(call,1000,1000)
    with pytest.raises(RuntimeError,match='candidate_cap'):
        ledger.reserve('repair',1000,1000)
    assert ledger.summary()['committed_usd']==pytest.approx(.002)


def test_native_run_with_only_rejected_proposals_reaches_the_phase_report(tmp_path,monkeypatch):
    import io
    import logging
    import os
    import urllib.request
    from pathlib import Path
    from types import SimpleNamespace
    pytest.importorskip('triton_kernel_agent')
    if not os.getenv('KERNEL_BASELINE_PREFLIGHT'): pytest.skip('requires registered public starts')
    from kernel_study import agent_campaign as campaign, public_referee as referee
    from kernel_study.public_cases import experiment
    from kernel_study.reference_report import compare
    from kernel_study.reference_search import Ledger, MeteredChat
    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import OptimizationOrchestrator

    class FailedProposals:
        def __init__(self,**options):
            self.directory=options['log_dir']
            self.directory.mkdir()
            (self.directory/'manager.log').write_text('no successful workers\n')

        def run_optimization(self,**options):
            worker=VerificationWorker(0,self.directory,self.directory,openai_model='test-model')
            obj=SimpleNamespace(logger=logging.getLogger('campaign-test'),artifact_dir=self.directory,verification_worker=worker)
            for number in (1,2):
                assert OptimizationOrchestrator._generate_optimized_kernel(obj,'make a proposal',number) is None
            return dict(success=False,total_rounds=2,kernel_code=None)

    def response(*args,**kwargs):
        return io.BytesIO(json.dumps({'model':'test-model','choices':[{'message':{'content':'No source this round.'}}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',response)
    monkeypatch.setattr(campaign,'native_manager',lambda:FailedProposals)
    measurement=dict(status='ok',candidate={'median_ms':1.},incumbent={'median_ms':1.},
                     compile={'median_ms':1.},compile_candidate={'median_ms':1.},paired_speedup=1.)
    monkeypatch.setattr(referee,'run_worker',lambda *args:measurement)
    output=tmp_path/'greedy'
    ledger=Ledger(tmp_path,'greedy',2,2,1.,1.,1.)
    provider=MeteredChat(ledger,output,model='test-model',endpoint='http://localhost:1')
    case=experiment()['cases']['4']
    result=campaign.run_native(case,11,{},provider,output,Path(os.environ['KERNEL_BASELINE_PREFLIGHT'])/'4',2)
    assert not result['upstream_success'] and result['status']=='complete'
    greedy=referee.audit(output,'greedy')
    assert greedy['fallback'] and not greedy['correct_new'] and greedy['budget']['attempts']==2
    report=compare([dict(greedy,agent='single'),greedy],[4],[11],{(4,11):measurement})
    assert report['paired_searches']==1 and report['fallbacks']['greedy']==1
