import pytest


def test_public_tasks_and_limits_are_stated_before_search():
    from kernel_study.public_cases import experiment
    plan = experiment()
    assert plan['development'] == [4, 23, 40]
    assert plan['holdout'] == [2, 26, 36]
    assert (plan['phases']['development']['candidate_cap'], plan['phases']['development']['call_cap']) == (4, 16)
    assert plan['cases']['40']['output_shape'] == [16, 64, 256, 256]
    assert plan['cases']['36']['output_shape'] == [112, 64, 512, 512]
    assert plan['cases']['23']['atol'] < 1/393216


def test_native_and_declarative_gelu_forward_share_device_code(tmp_path):
    from kernel_study.public_cases import experiment, start, load_native, render_native, local_forward
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
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


def test_common_referee_rejects_mutation_wrong_shape_and_wrong_dtype():
    from kernel_study.public_cases import check_forward
    torch = pytest.importorskip('torch')
    x = torch.tensor([2.])
    case = dict(atol=0.,rtol=0.)
    for bad in (lambda v:v.add_(1.), lambda v:(v+1).reshape(1,1), lambda v:(v+1).double()):
        with pytest.raises(AssertionError):
            check_forward(bad, [x.clone()], x+1, case)


def test_native_layernorm_exposes_affine_parameter_names():
    import ast
    from kernel_study.public_cases import experiment, render_native, start
    case = experiment()['cases']['40']
    module = ast.parse(render_native(case,*start(case)))
    assert [arg.arg for arg in module.body[-1].args.args] == ['X','weight','bias']


def test_hand_layernorm_preserves_the_three_normalized_dimensions():
    from kernel_study.public_preflight import hand_forward
    torch = pytest.importorskip('torch')
    pytest.importorskip('tritonbench')
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    x = torch.randn(3,2,4,8,device='cuda')
    w,b = (torch.randn_like(x[0]) for _ in range(2))
    result = hand_forward(40,[x,w,b])()
    torch.testing.assert_close(result,torch.nn.functional.layer_norm(x,x.shape[1:],w,b))


def test_layernorm_start_preserves_small_differences_on_large_offsets(tmp_path):
    from kernel_study import public_cases as public
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA required')
    case={**public.experiment()['cases']['40'],'output_shape':[3,4,16,79]}
    torch.manual_seed(10001)
    x=torch.rand(case['output_shape'],device='cuda')/32+1024
    model=torch.nn.LayerNorm(x.shape[1:],device='cuda').eval()
    with torch.no_grad():
        w,b=model.weight,model.bias
        source,config=public.start(case)
        path=tmp_path/'device.py'
        path.write_text(source)
        fn=public.local_forward(case,path,config,[x,w,b])
        actual=public.check_forward(lambda *_:fn(),[x,w,b],model(x),case,
                                    reference_chunks=public.fp64_chunks(model,[x]))
        assert actual['elements']==x.numel()
