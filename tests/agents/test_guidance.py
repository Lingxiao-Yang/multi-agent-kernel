from kernel_study.api_guidance import VERSION, guidance
from kernel_study.core import cases
import copy
from kernel_study.api_guidance import history_context


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
