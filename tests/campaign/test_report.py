import pytest

from kernel_study.campaign.report import compare, phase_rows
from kernel_study.records import write_json


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
                 candidates=[{'attempt':1,'usd':charge}])
        write_json(tmp_path/(name+suffix)/'audit.json',row)
    rows,attempts,_=phase_rows(tmp_path,'development',[4],[11])
    assert len(rows)==1 and len(attempts)==2
    assert rows[0]['total_committed_usd']==pytest.approx(.17)
    assert rows[0]['total_completed_usd']==pytest.approx(.05)
    assert rows[0]['total_attempts']==2
    assert rows[0]['matched_candidates'][0]['attempt']==2


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
