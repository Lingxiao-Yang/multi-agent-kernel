import pytest


def test_fallback_stays_in_ratio_and_valid_new_is_counted_independently():
    from kernel_study.reference_report import compare
    rows = []
    for case in [2,26,36]:
        for seed in [11,22,33]:
            for agent in ['single','greedy']:
                fallback = (case,seed,agent)==(2,11,'single')
                rows.append(dict(case=case,seed=seed,agent=agent,status='complete',fallback=fallback,
                                 correct_new=True,latency_ms=2.0 if fallback else 1.0))
    result = compare(rows,[2,26,36],[11,22,33])
    assert result['paired_searches']==9
    assert result['tasks']['2']['ratio'] == pytest.approx(2**(1/3))
    assert result['ratio']==pytest.approx(2**(1/9))
    assert result['correct_new']=={'single':9,'greedy':9}
    assert result['complete']
    rows.pop()
    assert not compare(rows,[2,26,36],[11,22,33])['complete']


def test_one_extra_fallback_across_threshold_requires_remeasurement():
    from kernel_study.reference_report import compare
    rows = [dict(case=2,seed=s,agent=a,status='complete',correct_new=True,
                 fallback=(a=='single' and s==11),latency_ms=2. if a=='single' and s==11 else 1.)
            for s in [11,22,33] for a in ['single','greedy']]
    assert compare(rows,[2],[11,22,33])['decision']=='pending-fallback-remeasurement'
