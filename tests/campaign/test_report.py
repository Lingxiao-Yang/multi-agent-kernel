from argparse import ArgumentParser, Namespace

import pytest

from kernel_study.campaign import report
from kernel_study.campaign.report import compare, phase_rows
from kernel_study.records import write_json


def test_matvec_slice_report_includes_the_two_model_free_reference_times(tmp_path):
    from kernel_study.cli import main
    from kernel_study.records import read_json
    main(['report','matvec-development',str(tmp_path)])
    result = read_json(tmp_path/'matvec-development-report.json')
    assert result['matvec_reference_ms']=={'random_best_of_four':4.10,'best_launch':3.17}
    assert list(result['tasks'])==['4'] and result['decision']=='incomplete'


def test_matching_greedy_can_still_miss_random_search_in_both_seeds(tmp_path):
    from kernel_study.cli import main
    from kernel_study.records import read_json
    for seed in (11,22):
        for agent in ('single','greedy'):
            directory=tmp_path/f'matvec-development-4-{agent}-{seed}'
            row=dict(agent=agent,case=4,seed=seed,status='complete',correct_new=True,fallback=False,
                     budget={'attempts':4},candidates=[],cost_usd=.1,initial_ms=10.,latency_ms=8.)
            write_json(directory/'audit.json',row)
        pair=tmp_path/'pairs'/f'matvec-development-4-{seed}'
        write_json(pair/'direct.result.json',dict(status='ok',candidate={'median_ms':8.},incumbent={'median_ms':8.}))
    main(['report','matvec-development',str(tmp_path)])
    result=read_json(tmp_path/'matvec-development-report.json')
    assert result['decision']=='matvec-slice-complete' and result['comparison_to_greedy']=='passes'
    assert result['at_or_below_random_four']=={a:{'count':0,'completed':2} for a in ('single','greedy')}


def test_round_report_matches_parent_sources_and_keeps_upstream_timeout_separate(tmp_path):
    from kernel_study.records import digest
    (tmp_path/'baseline.py').write_text('start')
    (tmp_path/'proposal_001.py').write_text('new')
    for i,source in [(1,'start'),(2,'new')]:
        worker=tmp_path/'greedy/workers/w0'/f'r{i}'
        (worker/'artifacts').mkdir(parents=True)
        (worker/'logs').mkdir()
        (worker/'artifacts/kernel_round_0.py').write_text(source)
        (worker/'logs/worker_0.log').write_text('t - w - ERROR - Test timed out\n' if i==2 else '')
    artifacts=tmp_path/'greedy/workers/w0/r1/artifacts'
    (artifacts/'kernel_round_1.py').write_text('new')
    write_json(artifacts/'benchmark_results.json',{'kernels':{'kernel_round_1':{'time_ms':8.}}})
    result=dict(status='ok',candidate={'median_ms':4.},incumbent={'median_ms':10.},confirmed=True)
    write_json(tmp_path/'audit_001.request.json',{'incumbent_source':'/cluster/baseline.py'})
    write_json(tmp_path/'audit_001.result.json',result)
    summary=dict(rounds=[{'round':1},{'round':2}],history=[dict(attempt=1,source_hash=digest('new'),result=result)])
    data=report.greedy_trajectory(tmp_path,summary)
    first,second=data['rounds']
    assert first['upstream_candidates'][0] == dict(attempt=1,upstream_ms=8.,public_ms=4.)
    assert second['parent']=='proposal_001.py' and second['parent_change']['confirmed']
    assert second['parent_change']['parent_ms']==10. and second['parent_change']['candidate_ms']==4.
    assert [r['upstream_candidate_timeouts'] for r in data['rounds']]==[0,1]
    assert data['upstream_initial_timeouts'] is None


def test_matched_smoke_can_be_reported_through_cli_without_a_gpu(tmp_path):
    from kernel_study.cli import main
    from kernel_study.records import read_json
    main(['report','matched-smoke',str(tmp_path)])
    result=read_json(tmp_path/'matched-smoke-report.json')
    assert list(result['tasks'])==['4'] and result['decision']=='incomplete'


def test_round_report_keeps_partial_archives_and_ambiguous_sources_unassigned(tmp_path):
    from kernel_study.records import digest
    artifacts=tmp_path/'greedy/workers/w0/r1/artifacts'
    artifacts.mkdir(parents=True)
    (artifacts/'kernel_round_0.py').write_text('repeated')
    (artifacts/'kernel_round_1.py').write_text('repeated')
    write_json(artifacts/'benchmark_results.json',{'kernels':{'kernel_round_1':{'time_ms':7.}}})
    entries=[dict(attempt=i,source_hash=digest('repeated'),result={}) for i in (1,2)]
    summary=dict(rounds=[{'round':1}],history=entries)
    point=report.greedy_trajectory(tmp_path,summary)['rounds'][0]
    assert point['parent'] is None and point['upstream_candidates']==[]
    assert point['upstream_candidate_timeouts'] is None


def test_rejection_reasons_are_counted_per_agent():
    rows = [dict(case=4,seed=11,agent=a,status='complete',latency_ms=1.,correct_new=False,
                 fallback=True,candidates=[{'status':s} for s in statuses])
            for a,statuses in [('single',['static_rejected','incorrect']),('greedy',['static_rejected','compile_error','ok'])]]
    reasons = compare(rows,[4],[11],{})['rejections']
    assert reasons['single']['static_check'] == reasons['single']['numerical'] == 1
    assert reasons['greedy']['static_check'] == reasons['greedy']['compilation'] == 1


def test_report_separates_gpu_timeouts_and_confirmed_gains():
    rows=[{'status':'timeout'},{'status':'ok','selected':True},{'status':'ok','selected':False}]
    counts=report.rejection_counts(rows)
    assert counts['timeout']==counts['confirmed']==counts['unconfirmed']==1


def test_report_remeasures_only_pairs_assigned_to_its_card(tmp_path,monkeypatch):
    monkeypatch.setattr(report,'experiment',lambda: {'development':[4],'development_seeds':[11,22]})
    monkeypatch.setattr(report,'check_device',lambda plan: {'uuid':'GPU-a'})
    write_json(tmp_path/'development-protocol.json', {'code_hashes':report.code_hashes()})
    for seed,uuid in [(11,'GPU-a'),(22,'GPU-b')]:
        write_json(tmp_path/'pairs'/f'development-4-{seed}'/'device.json',{'uuid':uuid})
    measured = []
    monkeypatch.setattr(report,'measure_pair',lambda dest,*sides: measured.append(dest.name))
    args = Namespace(output=tmp_path,phase='development',measure_pairs=True,freeze=False)
    report.run_report(args,ArgumentParser())
    assert measured == ['development-4-11']


def test_primary_ratio_uses_only_the_direct_pair():
    rows = [dict(case=40, seed=11, agent=a, correct_new=True, fallback=False, status='complete',
                 latency_ms=t) for a,t in [('single',9.), ('greedy',1.)]]
    direct = {(40,11): {'status':'ok', 'candidate':{'median_ms':1.04}, 'incumbent':{'median_ms':1.}}}
    assert compare(rows, [40], [11], direct)['ratio'] == pytest.approx(1.04)
    assert not compare(rows, [40], [11], {})['complete']


def test_retry_report_charges_both_runs_but_counts_one_search(tmp_path):
    name = 'development-4-single-11'
    for suffix, status, charge in [('', 'interrupted',.12),('-retry1','complete',.05)]:
        row=dict(agent='single',case=4,seed=11,status=status,stop_reason='transport' if not suffix else 'candidate_cap',
                 budget={'attempts':1},cost_usd=0. if not suffix else charge,committed_usd=charge,
                 candidates=[{'attempt':1,'usd':charge,'status':'static_rejected' if not suffix else 'ok',
                              'selected':bool(suffix),'endpoint_status':'unconfirmed' if suffix else None}])
        write_json(tmp_path/(name+suffix)/'audit.json',row)
    rows,attempts,_=phase_rows(tmp_path,'development',[4],[11])
    assert len(rows)==1 and len(attempts)==2
    assert rows[0]['total_committed_usd']==pytest.approx(.17)
    assert rows[0]['total_completed_usd']==pytest.approx(.05)
    assert rows[0]['total_attempts']==2
    assert rows[0]['matched_candidates'][0]['attempt']==2
    rejected = compare(rows,[4],[11],{})['rejections']['single']
    assert rejected['static_check'] == rejected['unconfirmed'] == 1


def pairs(rows):
    return {(r['case'],r['seed']):dict(status='ok',candidate={'median_ms':r['latency_ms']},incumbent={'median_ms':1.})
            for r in rows if r['agent']=='single'}


def test_fallback_stays_in_ratio_and_valid_new_is_counted_independently():
    rows = []
    for case in [2,26,36]:
        for seed in [11,22,33]:
            for agent in ['single','greedy']:
                fallback = (case,seed,agent)==(2,11,'single')
                rows.append(dict(case=case,seed=seed,agent=agent,status='complete',fallback=fallback,
                                 correct_new=True,latency_ms=2.0 if fallback else 1.0))
    result = compare(rows,[2,26,36],[11,22,33],pairs(rows))
    assert result['paired_searches']==9
    assert result['tasks']['2']['ratio'] == pytest.approx(2**(1/3))
    assert result['ratio']==pytest.approx(2**(1/9))
    assert result['correct_new']=={'single':9,'greedy':9}
    assert result['complete']
    rows.pop()
    assert not compare(rows,[2,26,36],[11,22,33],pairs(rows))['complete']


def test_one_extra_fallback_across_threshold_requires_remeasurement():
    rows = [dict(case=2,seed=s,agent=a,status='complete',correct_new=True,
                 fallback=(a=='single' and s==11),latency_ms=2. if a=='single' and s==11 else 1.)
            for s in [11,22,33] for a in ['single','greedy']]
    assert compare(rows,[2],[11,22,33],pairs(rows))['decision']=='pending-fallback-remeasurement'
