import pytest
from kernel_study.core import digest, write_json
from kernel_study.core import cases
from kernel_study.core import baseline_config, validate_config
import json
from kernel_study.core import baseline_config
from kernel_study.providers import ScriptedProvider
from kernel_study.core import launch_contract
from kernel_study.metrics import throughput, work_and_traffic
from kernel_study.metrics import throughput
from kernel_study import metrics
from kernel_study.core import VARIANTS, write_json
from kernel_study.core import ROOT, read_json, write_json
from kernel_study.repair_analysis import summarize
from kernel_study.core import ROOT, read_json
import copy


def test_replay_keeps_fallbacks_and_checks_archived_source(tmp_path):
    from kernel_study.replay import archived_entries
    path = tmp_path / 'repair-strong-3584057' / 'campaign' / 'layernorm_irregular_single_22'
    path.mkdir(parents=True)
    source = 'baseline'
    (path / 'baseline.py').write_text(source)
    config = {'launches': []}
    write_json(path / 'baseline.request.json', {'config': config})
    write_json(path / 'summary.json', {
        'manifest': {'case': {'id':'layernorm_irregular'}, 'baseline_hash': digest(source)},
        'final': {'candidate_id': 0, 'config': config, 'result': {'status': 'ok'}}, 'history': []})
    entries = list(archived_entries(tmp_path))
    assert [entry['role'] for entry in entries] == ['start', 'final']
    assert entries[1]['candidate_id'] == 0
    assert entries[0]['key'] == entries[1]['key']
    (path / 'baseline.py').write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        list(archived_entries(tmp_path))


def test_case_matrix_has_four_families_three_cases_and_explicit_tolerance():
    suite = cases()
    assert len(suite)==12
    for f in ['matmul','softmax','layernorm','elementwise']:
        assert sum(c['family']==f for c in suite)==3
    assert len({c['id'] for c in suite})==12
    assert all(c['atol']>0 and c['rtol']>0 for c in suite)


def test_config_rejects_silent_extra_knobs_and_bool_as_int():
    c = baseline_config('matmul')
    assert validate_config(c)==c
    with pytest.raises(ValueError):
        validate_config({**c,'autotune':True})
    with pytest.raises(ValueError):
        validate_config({**c,'num_warps':True})


def test_scripted_control_has_zero_real_model_tokens():
    p = ScriptedProvider()
    reply = p.complete('',{'incumbent':{'config':baseline_config('softmax')},'attempt':1,'case':{'family':'softmax'}},'implement',11)
    assert reply.input_tokens==reply.output_tokens==0
    assert reply.metadata['synthetic']
    assert json.loads(reply.text)['config']['fused']


def test_softmax_contract_exposes_ignored_tensor_knobs():
    contract=launch_contract('softmax')
    assert 'tensor_core' in contract['ignored_config_fields']
    assert 'block_m' in contract['ignored_config_fields']
    assert 'fused' in contract['active_config_fields']
    assert 'tensor_core' in launch_contract('matmul')['active_config_fields']


def test_matmul_counts_and_units():
    case={'family':'matmul','shape':[10,20,30],'dtype':'float16'}
    assert work_and_traffic(case)=={'flops':12000,'minimum_bytes':2200}
    r=throughput(case,1,peak_gbps=10,peak_tflops=1)
    assert r['logical_gbps']==pytest.approx(.0022)
    assert r['achieved_tflops']==pytest.approx(.000012)


def test_no_unverified_roofline_fraction():
    r=throughput({'family':'softmax','shape':[10,20],'dtype':'float32'},1)
    assert r['minimum_bytes']==1600
    assert r['fraction_peak_bandwidth'] is None
    assert r['fraction_peak_compute'] is None


def test_archived_cross_round_comparison_is_identified():
    old = {'candidate': {'median_ms': 2.}, 'library': {'median_ms': 12.}}
    report = metrics.library_timings(old, 1)
    assert report['library_over_candidate'] == 6.
    assert report['library_candidate_ms'] is None
    assert report['library_timing_basis'] == 'historical-unpaired'
    with pytest.raises(KeyError):
        metrics.library_timings(old, 2)


def _matrix(root,plan):
    for case in plan['cases']:
        for v in VARIANTS:
            for seed in plan['seeds']:
                history=[dict(attempt=i,result={'status':'ok' if i==3 else 'incorrect'},accepted=i==3) for i in range(1,9)]
                write_json(root/f'{case}_{v}_{seed}'/'summary.json',dict(status='complete',
                    manifest={'case':{'id':case},'variant':v,'seed':seed},history=history,
                    budget=dict(attempts=8,calls=32,tokens=123),final={'candidate_id':3},speedup=1.5))


def test_repair_prefixes_readiness_and_incomplete_matrix(tmp_path):
    plan=read_json(ROOT/'configs/repair_round.json');root=tmp_path/'runs'
    _matrix(root,plan)
    report=summarize(root,tmp_path/'report.json',plan)
    assert report['complete'] and not report['stronger_required']
    assert report['first_correct_after_attempt_2']==16
    assert report['searches_with_recovery']==16
    assert report['runs'][0]['prefixes']['2']['correct']==0
    assert report['runs'][0]['prefixes']['4']['correct']==1
    for path in list(root.glob('*/summary.json'))[:3]:
        data=read_json(path);data['final']['candidate_id']=0;write_json(path,data)
    report=summarize(root,tmp_path/'report.json',plan)
    assert report['stronger_required']
    next(root.glob('*/summary.json')).unlink()
    report=summarize(root,tmp_path/'report.json',plan)
    assert not report['complete'] and report['stronger_required'] is None


def test_all_reserved_round_caps_fit_approved_total():
    p=read_json(ROOT/'configs/repair_round.json')
    assert p['prior_completed_usd']+p['local_smoke_cap_usd']+p['mini']['cap_usd']+p['strong']['cap_usd']<=p['approved_total_usd']
    assert p['attempts']==8 and p['calls_per_search']==32


def test_cohort_allows_family_baselines_but_rejects_within_family_drift():
    from kernel_study.analysis import validate_cohort
    a={'manifest':{'protocol_hash':'p','knowledge_hash':'k','provider':'llm','model':'m',
                  'environment':{'gpu':'gpu'},'case':{'family':'softmax','id':'s'},
                  'case_hash':'s','baseline_hash':'softmax-source'},'resolved_model':'m'}
    b=copy.deepcopy(a)
    b['manifest'].update(case={'family':'matmul','id':'m'},case_hash='m',baseline_hash='matmul-source')
    validate_cohort([a,b])
    drift=copy.deepcopy(a);drift['manifest']['baseline_hash']='changed'
    with pytest.raises(ValueError,match='Incompatible'): validate_cohort([a,b,drift])
    drift=copy.deepcopy(a);drift['manifest']['case_hash']='changed'
    with pytest.raises(ValueError,match='Incompatible'): validate_cohort([a,b,drift])
    drift=copy.deepcopy(a);drift['manifest']['code_hashes']={'kernel_study/search.py':'changed'}
    with pytest.raises(ValueError,match='Incompatible'): validate_cohort([a,b,drift])
