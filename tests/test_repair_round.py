import copy
from kernel_study.api_guidance import VERSION, guidance, history_context
from kernel_study.core import ROOT, VARIANTS, cases, read_json, write_json
from kernel_study.repair_analysis import summarize


def test_guidance_resolves_irregular_shapes_without_optimized_code():
    for case in cases():
        result=guidance(case);s=result['resolved_scalars']
        assert s['N']==case['shape'][1]
        assert s['TOTAL']==case['shape'][0]*case['shape'][1]
        assert s['RB']>=s['N'] and s['RB']&(s['RB']-1)==0
        assert result['version']==VERSION
        assert '@triton.jit' not in str(result)


def test_context_keeps_diagnostics_but_does_not_repeat_generated_review_source():
    original=[dict(attempt=2,result=dict(status='incorrect',error='wrong mask'),
                   review=dict(source='large regenerated module',diagnosis='cover full row'),
                   source_hash='hash',accepted=False)]
    before=copy.deepcopy(original)
    slim=history_context(original)
    assert original==before
    assert slim[0]['result']['error']=='wrong mask'
    assert slim[0]['review']=={'diagnosis':'cover full row'}
    assert slim[0]['source_hash']=='hash'


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
