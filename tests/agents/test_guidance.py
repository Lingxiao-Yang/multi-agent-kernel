import copy

from kernel_study.agents.guidance import VERSION, guidance, history_context
from kernel_study.tasks.kernelbench import experiment


def test_guidance_resolves_irregular_shapes_without_optimized_code():
    for case in experiment()['cases'].values():
        result=guidance(case)
        assert result['output_shape']==case['output_shape']
        assert 'resolved_scalars' not in result
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
