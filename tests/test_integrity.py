import json
from pathlib import Path
import numpy as np
import pytest
from kernel_study.core import ROOT, Budget, baseline_config, cases, validate_config
from kernel_study.source_policy import check_source
from kernel_study.knowledge import select_cards
from kernel_study.analysis import geometric_mean, paired_effects
from kernel_study.providers import ScriptedProvider


def test_candidate_failure_and_repair_consume_budget():
    b = Budget(candidate_cap=2)
    b.reserve_attempt()  # failed compile
    b.reserve_attempt()  # attempted repair
    with pytest.raises(RuntimeError,match='candidate_cap'):
        b.reserve_attempt()


def test_calls_and_actual_tokens_are_charged_separately():
    b = Budget(call_cap=2,token_safety_cap=100)
    b.reserve_call(); b.charge_tokens(70,31)
    assert b.stop_reason() == 'token_safety_cap'
    assert b.tokens == 101  # actual usage is retained, never truncated to cap
    with pytest.raises(RuntimeError):
        b.reserve_call()


def test_call_cap_blocks_further_network_requests():
    b = Budget(call_cap=1)
    b.reserve_call()
    with pytest.raises(RuntimeError):
        b.reserve_call()


def test_baseline_source_passes_policy():
    check_source((ROOT/'kernel_study/kernels.py').read_text())


@pytest.mark.parametrize('source',[
    'import os\nos.remove("validator.py")',
    'import triton\nx = open("secret")',
    'import triton\n@triton.jit\ndef f(X):\n    eval("1")',
    'import triton\n@triton.jit\ndef f(X):\n    x = X.__class__',
    'import triton\n@triton.autotune([])\ndef f(X):\n    pass',
    'import triton\n@triton.jit\ndef f(X=open("x")):\n    pass',
    'import triton\n@triton.jit\ndef f(X):\n    import os',
])
def test_device_policy_rejects_host_execution(source):
    with pytest.raises(ValueError):
        check_source(source)


def test_case_matrix_has_four_families_three_cases_and_explicit_tolerance():
    suite = cases()
    assert len(suite)==12
    for f in ['matmul','softmax','layernorm','elementwise']:
        assert sum(c['family']==f for c in suite)==3
    assert len({c['id'] for c in suite})==12
    assert all(c['atol']>0 and c['rtol']>0 for c in suite)


def test_knowledge_selection_is_shared_deterministic_and_ablatable():
    a = select_cards('softmax')
    assert a == select_cards('softmax')
    assert any(c['category']=='fusion' for c in a)
    assert all(c['category']!='fusion' for c in select_cards('softmax',excluded_category='fusion'))
    assert {c['id'] for c in a} == {c['id'] for c in select_cards('softmax',{'kernels':[{'spills':8}]})}


def test_config_rejects_silent_extra_knobs_and_bool_as_int():
    c = baseline_config('matmul')
    assert validate_config(c)==c
    with pytest.raises(ValueError):
        validate_config({**c,'autotune':True})
    with pytest.raises(ValueError):
        validate_config({**c,'num_warps':True})


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


def test_scripted_control_has_zero_real_model_tokens():
    p = ScriptedProvider()
    reply = p.complete('',{'incumbent':{'config':baseline_config('softmax')},'attempt':1,'case':{'family':'softmax'}},'implement',11)
    assert reply.input_tokens==reply.output_tokens==0
    assert reply.metadata['synthetic']
    assert json.loads(reply.text)['config']['fused']
