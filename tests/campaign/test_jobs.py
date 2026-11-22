import pytest
from kernel_study.core import read_json, write_json
from kernel_study.core import read_json


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


def test_reporting_only_revision_preserves_the_registered_search(tmp_path):
    from kernel_study.agent_campaign import register
    registered=dict(revision='before',code_hashes={'worker.py':'abc'},jobs=[[4,'greedy',11]])
    register(tmp_path,'development-protocol',registered)
    register(tmp_path,'development-protocol',dict(registered,revision='report-only'))
    assert read_json(tmp_path/'development-protocol.json')==registered
    with pytest.raises(ValueError):
        register(tmp_path,'development-protocol',dict(registered,revision='worker-changed',code_hashes={'worker.py':'def'}))


def test_campaign_selects_only_registered_jobs_and_separate_calibration():
    from kernel_study.agent_campaign import jobs
    from kernel_study.public_cases import experiment
    plan = experiment()
    assert len(jobs(plan,'development')) == 12
    assert len(jobs(plan,'holdout')) == 18
    assert jobs(plan,'smoke') == [(4, 'greedy', 771)]
    assert jobs(plan,'calibration') == [(4, 'greedy', 772)]
    for i in range(0,12,2):
        pair = jobs(plan,'development')[i:i+2]
        assert {p[1] for p in pair} == {'single','greedy'}
        assert len({(p[0],p[2]) for p in pair}) == 1
