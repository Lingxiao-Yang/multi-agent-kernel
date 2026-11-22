import pytest

from kernel_study.campaign.factorial import geometric_mean, paired_effects


def test_known_factorial_interaction_and_no_improvement_runs():
    rows = [{'case':case,'seed':seed,'variant':v,'speedup':speedup}
            for case in ['a','b','c'] for seed in [11,22,33]
            for v,speedup in [('single',1),('multi',2),('single_knowledge',3),('multi_knowledge',12)]]
    r = paired_effects(rows,resamples=100)
    assert r['complete_pairs']==9
    assert r['bootstrap_seed']==42
    assert r['effects']['interaction_ratio']['ratio']==pytest.approx(2)
    assert geometric_mean([1,4])==pytest.approx(2)
    with pytest.raises(ValueError):
        paired_effects(rows+[rows[0]])


def test_missing_arms_cannot_support_factorial_conclusions():
    assert paired_effects([{'case':'a','seed':11,'variant':'single','speedup':1}])['complete_pairs']==0
