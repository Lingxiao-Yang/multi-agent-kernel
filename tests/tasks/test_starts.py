import pytest
import ast
from kernel_study.public_cases import experiment, start
from kernel_study.source_space import validate_plan, _body_shapes


@pytest.mark.gpu
def test_native_and_declarative_gelu_forward_share_device_code(tmp_path):
    from kernel_study.public_cases import experiment, start, load_native, render_native, local_forward
    import torch
    case = {**experiment()['cases']['26'], 'output_shape': [13, 781]}
    source, config = start(case)
    device = tmp_path/'device.py'
    device.write_text(source)
    host = tmp_path/'native.py'
    host.write_text(render_native(case,source,config))
    x = torch.randn(13,781,device='cuda')
    saved = x.clone()
    native = load_native(host)
    local = local_forward(case,device,config,[x])
    assert torch.equal(local(),native(x))
    torch.testing.assert_close(local(),torch.nn.functional.gelu(x))
    assert torch.equal(saved,x)


def test_native_layernorm_exposes_affine_parameter_names():
    import ast
    from kernel_study.public_cases import experiment, render_native, start
    case = experiment()['cases']['40']
    module = ast.parse(render_native(case,*start(case)))
    assert [arg.arg for arg in module.body[-1].args.args] == ['X','weight','bias']


def test_public_starts_read_their_own_device_modules(monkeypatch):
    from kernel_study import public_cases as p
    original=p.device_function
    def read(path,name):
        assert path.name != 'kernels.py'
        return original(path,name)
    monkeypatch.setattr(p,'device_function',read)
    for case in ('2','23','40'):
        assert p.start(p.experiment()['cases'][case])[1]['launches']


def test_only_baseline_functions_are_exposed_and_all_launches_match():
    for number in ('2', '23', '40', '26', '36'):
        source, config = start(experiment()['cases'][number])
        validate_plan(config)
        _body_shapes(source, config)
        names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        assert names == {item['kernel'] for item in config['launches']}
