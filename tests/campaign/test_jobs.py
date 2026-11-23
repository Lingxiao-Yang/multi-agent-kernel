import pytest
from argparse import ArgumentParser, Namespace

from kernel_study.campaign import jobs as campaign
from kernel_study.campaign.jobs import attempt_path, jobs, register
from kernel_study.records import code_hashes, digest, read_json, write_json
from kernel_study.tasks.kernelbench import experiment


def test_only_one_new_attempt_after_transport_failure(tmp_path):
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


def test_frozen_code_and_phase_settings():
    plan=experiment()
    frozen=code_hashes()
    assert {'kernel_study/judge/select.py', 'kernel_study/tasks/baselines/gelu.py',
            'kernel_study/campaign/jobs.py', 'kernel_study/campaign/preflight.py',
            'configs/protocol.json', 'configs/agent_baseline.json', 'knowledge/cards.json'} <= frozen.keys()
    assert 'kernel_study/campaign/report.py' not in frozen
    assert 'kernel_study/campaign/factorial.py' not in frozen
    assert plan['gpu_specs']['peak_memory_bw_gbps'] == 1555
    assert plan['gpu'] == 'NVIDIA A100 PCIe 40GB'
    assert [(v['candidate_cap'],v['call_cap'],v['rounds']) for v in
            (plan['phases'][k] for k in ('smoke','calibration','development','holdout'))] == [
                (1,4,1),(32,128,8),(4,16,4),(4,16,4)]


def test_reporting_only_revision_preserves_the_registered_search(tmp_path):
    registered=dict(revision='before',code_hashes={'worker.py':'abc'},jobs=[[4,'greedy',11]])
    register(tmp_path,'development-protocol',registered)
    register(tmp_path,'development-protocol',dict(registered,revision='report-only'))
    assert read_json(tmp_path/'development-protocol.json')==registered
    with pytest.raises(ValueError):
        register(tmp_path,'development-protocol',dict(registered,revision='worker-changed',code_hashes={'worker.py':'def'}))


def test_campaign_selects_only_registered_jobs_and_separate_calibration():
    plan = experiment()
    assert len(jobs(plan,'development')) == 12
    assert len(jobs(plan,'holdout')) == 18
    assert jobs(plan,'smoke') == [(4, 'single', 771)]
    assert jobs(plan,'calibration') == [(4, 'greedy', 772)]
    for i in range(0,12,2):
        pair = jobs(plan,'development')[i:i+2]
        assert {p[1] for p in pair} == {'single','greedy'}
        assert len({(p[0],p[2]) for p in pair}) == 1


def test_native_smoke_and_reference_search_are_independent(tmp_path,monkeypatch):
    plan = experiment()
    base = tmp_path/'starts'/'4'
    base.mkdir(parents=True)
    (base/'input.py').write_text('native source')
    problem = tmp_path/'problem.py'
    problem.write_text('task source')
    identity = dict(native_source_hash=digest('native source'),problem_hash=digest('task source'))
    write_json(base/'preflight.json',dict(identity,checks={'native':{}}))
    write_json(base/'stress.json',dict(identity,checks={'native':[{}]}))
    write_json(tmp_path/'counters.json',{'counter_access_verified':True})
    args = Namespace(phase='smoke',job=0,preflight=base.parent,output=tmp_path/'campaign',
                     counters=tmp_path/'counters.json',model='test-model',endpoint='http://localhost:1/v1',
                     input_price=2.,output_price=8.,retry_transport=False)
    gpu = {'gpu':plan['gpu'],'driver':plan['driver'],'uuid':'GPU-first','node':'node-1'}
    monkeypatch.setenv('OPENAI_API_KEY','test-value')
    monkeypatch.setattr(campaign,'check_device',lambda plan: dict(gpu))
    monkeypatch.setattr(campaign.kernelbench,'upstream',lambda name: tmp_path)
    monkeypatch.setattr(campaign.kernelbench,'task_path',lambda case: problem)
    monkeypatch.setattr(campaign.starts,'start',lambda case: 'native source')
    monkeypatch.setattr(campaign.kernelbench,'test_source',lambda case: 'verification script')
    calls, pairs = [], []
    def record(agent,protocol):
        calls.append((agent,protocol['improvement_threshold'],protocol['candidate_cap'],protocol['call_cap']))
        return {'status':'complete'}
    monkeypatch.setattr(campaign,'run_native',lambda case,seed,protocol,*rest: record('greedy',protocol))
    monkeypatch.setattr(campaign,'search',lambda case,agent,seed,protocol,*rest: record(agent,protocol))
    monkeypatch.setattr(campaign,'audit',lambda *args: None)
    monkeypatch.setattr(campaign,'measure_pair',lambda dest,a,b: pairs.append((dest.name,a.name,b.name)))
    campaign.run_campaign(args,ArgumentParser())
    args.phase = 'calibration'
    gpu['uuid'] = 'GPU-second'
    campaign.run_campaign(args,ArgumentParser())
    assert calls == [('single',1.01,1,4),('greedy',1.01,32,128)]
    assert not pairs
    assert plan['total_budget_usd'] - plan['historical_cost_usd'] == pytest.approx(11.1973984)


def test_matched_smoke_is_one_candidate_for_each_agent_on_the_same_seed():
    plan = experiment()
    assert campaign.jobs(plan,'matched-smoke') == [(4,'greedy',771),(4,'single',771)]
    phase = plan['phases']['matched-smoke']
    assert (phase['rounds'],phase['candidate_cap'],phase['call_cap'])==(1,1,4)


def test_paired_evidence_requires_the_frozen_source_and_the_actual_upstream_prompt(tmp_path):
    from kernel_study.records import event
    from kernel_study.campaign.jobs import shared_start_metrics
    src=tmp_path/'initial.py'
    src.write_text('frozen kernel')
    metrics={'example':{'memory_metric':87.1}}
    write_json(tmp_path/'greedy/round001_ncu_metrics.json', {'metadata':{'kernel_file':str(src)},'metrics':metrics})
    event(tmp_path/'model_events.jsonl','request',stage='analysis',messages=[{'content':str(metrics['example'])}])
    assert shared_start_metrics(tmp_path, 'frozen kernel')==metrics
    with pytest.raises(ValueError,match='source'):
        shared_start_metrics(tmp_path, 'different kernel')
    (tmp_path/'model_events.jsonl').write_text('')
    with pytest.raises(ValueError,match='prompt'):
        shared_start_metrics(tmp_path, 'frozen kernel')


def test_matched_prompt_audit_finds_one_changed_measurement(tmp_path):
    from kernel_study.records import event
    from kernel_study.campaign.jobs import check_single_evidence
    evidence={'metrics':{'k':{'counter':87.1}},'roofline':{'bottleneck':'memory'}}
    specs={'sm_count':108}
    import json
    for stage in ('analyze','advise','implement','review'):
        context={'incumbent':{'result':{'profile':{'ncu':evidence}}},'hardware':{'specs':specs}}
        event(tmp_path/'model_events.jsonl','request',stage=stage,messages=[{'role':'user','content':json.dumps(context)}])
    assert check_single_evidence(tmp_path,evidence,specs)['identical']
    assert not check_single_evidence(tmp_path,dict(evidence,metrics={'k':{'counter':40}}),specs)['identical']
