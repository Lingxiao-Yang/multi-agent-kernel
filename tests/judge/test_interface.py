import pytest
from kernel_study import bench
from kernel_study.core import cases
from kernel_study.source_space import baseline, validate_plan
from kernel_study.source_space import baseline, _body_shapes


@pytest.mark.gpu
@pytest.mark.parametrize('mode', ['config', 'source-v1'])
def test_timed_forward_returns_fresh_output(mode, tmp_path):
    import torch
    from kernel_study import source_space
    from kernel_study.core import baseline_config, cases
    case = next(c for c in cases() if c['id'] == 'softmax_irregular')
    inputs = bench.make_inputs(case, 71)
    config, path = baseline_config('softmax'), None
    if mode == 'source-v1':
        source, config = source_space.baseline(case)
        path = tmp_path / 'baseline.py'
        path.write_text(source)
    forward, _ = bench.prepare(case, config, inputs, path, fresh_output=True)
    first, second = forward(), forward()
    assert first.data_ptr() != second.data_ptr()
    for output in (first, second):
        torch.testing.assert_close(output, bench.reference(case, inputs), atol=case['atol'], rtol=case['rtol'])


@pytest.mark.parametrize('field,value', [('grid',[0]), ('grid',['__import__("os")']), ('args',['__class__']), ('num_warps',True)])
def test_launch_interface_does_not_evaluate_expressions(field,value):
    _,config=baseline(cases()[0])
    config['launches'][0][field]=value
    with pytest.raises(ValueError): validate_plan(config)


def test_wrong_arity_fails_before_gpu():
    source,config=baseline(cases()[0])
    config['launches'][0]['args'].pop()
    with pytest.raises(ValueError,match='argument count'): _body_shapes(source,config)
