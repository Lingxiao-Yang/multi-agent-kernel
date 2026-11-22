import pytest

from kernel_study.public_cases import start
from kernel_study.interface import launched_functions, prepare, validate_plan


@pytest.mark.gpu
def test_timed_forward_returns_fresh_output(public_case, tmp_path):
    import torch
    x = torch.randn(public_case['output_shape'], device='cuda')
    source, config = start(public_case)
    path = tmp_path/'baseline.py'
    path.write_text(source)
    forward, _ = prepare(public_case, config, [x], path, fresh_output=True)
    first, second = forward(), forward()
    assert first.data_ptr() != second.data_ptr()
    for output in (first, second):
        torch.testing.assert_close(output, torch.nn.functional.gelu(x), atol=public_case['atol'], rtol=public_case['rtol'])


@pytest.mark.parametrize('field,value', [('grid',[0]), ('grid',['__import__("os")']), ('args',['__class__']), ('num_warps',True)])
def test_launch_interface_does_not_evaluate_expressions(public_case, field, value):
    _, config = start(public_case)
    config['launches'][0][field] = value
    with pytest.raises(ValueError):
        validate_plan(config)


def test_wrong_arity_fails_before_gpu(public_case):
    source, config = start(public_case)
    config['launches'][0]['args'].pop()
    with pytest.raises(ValueError, match='argument count'):
        launched_functions(source, config)
