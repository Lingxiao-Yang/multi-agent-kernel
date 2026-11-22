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
    assert {'kernel_study/judge/select.py', 'kernel_study/tasks/kernels.py',
            'kernel_study/campaign/jobs.py', 'kernel_study/campaign/preflight.py',
            'configs/protocol.json', 'configs/agent_baseline.json', 'knowledge/cards.json'} <= frozen.keys()
    assert 'kernel_study/campaign/report.py' not in frozen
    assert 'kernel_study/campaign/factorial.py' not in frozen
    assert plan['gpu_specs']['peak_memory_bw_gbps'] == 1555
    assert plan['gpu'] == 'NVIDIA A100 PCIe 40GB'
    assert [(v['candidate_cap'],v['call_cap'],v['rounds']) for v in
            (plan['phases'][k] for k in ('smoke','calibration','development','holdout'))] == [
                (1,16,1),(32,128,8),(4,16,4),(4,16,4)]


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
    assert jobs(plan,'smoke') == [(4, 'greedy', 771),(4, 'single', 771)]
    assert jobs(plan,'calibration') == [(4, 'greedy', 772)]
    for i in range(0,12,2):
        pair = jobs(plan,'development')[i:i+2]
        assert {p[1] for p in pair} == {'single','greedy'}
        assert len({(p[0],p[2]) for p in pair}) == 1


def test_smoke_uses_only_its_case_and_one_physical_gpu(tmp_path,monkeypatch):
    plan = experiment()
    base = tmp_path/'starts'/'4'
    base.mkdir(parents=True)
    (base/'input.py').write_text('native source')
    problem = tmp_path/'problem.py'
    problem.write_text('task source')
    launch = {'mode':'test'}
    identity = dict(native_source_hash=digest('native source'),device_source_hash=digest('device source'),
                    problem_hash=digest('task source'),launch=launch)
    write_json(base/'preflight.json',dict(identity,identical_outputs=True))
    write_json(base/'stress.json',dict(identity,checks={'native':[{}],'declarative':[{}]}))
    write_json(tmp_path/'counters.json',{'counter_access_verified':True})
    args = Namespace(phase='smoke',job=0,preflight=base.parent,output=tmp_path/'campaign',
                     counters=tmp_path/'counters.json',model='test-model',endpoint='http://localhost:1/v1',
                     input_price=2.,output_price=8.,retry_transport=False)
    gpu = {'gpu':plan['gpu'],'driver':plan['driver'],'uuid':'GPU-first','node':'node-1'}
    monkeypatch.setenv('OPENAI_API_KEY','test-value')
    monkeypatch.setattr(campaign,'check_device',lambda plan: dict(gpu))
    monkeypatch.setattr(campaign.kernelbench,'upstream',lambda name: tmp_path)
    monkeypatch.setattr(campaign.kernelbench,'task_path',lambda case: problem)
    monkeypatch.setattr(campaign.starts,'start',lambda case: ('device source',launch))
    calls, pairs = [], []
    def record(agent,protocol):
        calls.append((agent,protocol['improvement_threshold'],protocol['candidate_cap'],protocol['call_cap']))
        return {'status':'complete'}
    monkeypatch.setattr(campaign,'run_native',lambda case,seed,protocol,*rest: record('greedy',protocol))
    monkeypatch.setattr(campaign,'search',lambda case,agent,seed,protocol,*rest: record(agent,protocol))
    monkeypatch.setattr(campaign,'audit',lambda *args: None)
    monkeypatch.setattr(campaign,'measure_pair',lambda dest,a,b: pairs.append((dest.name,a.name,b.name)))
    campaign.run_campaign(args,ArgumentParser())
    args.job = 1
    gpu['uuid'] = 'GPU-second'
    with pytest.raises(SystemExit):
        campaign.run_campaign(args,ArgumentParser())
    gpu['uuid'] = 'GPU-first'
    campaign.run_campaign(args,ArgumentParser())
    assert calls == [('greedy',1.01,1,16),('single',1.01,1,4)]
    assert pairs[-1] == ('smoke-4-771','smoke-4-single-771','smoke-4-greedy-771')
