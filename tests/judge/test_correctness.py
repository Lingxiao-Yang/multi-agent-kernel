import pytest


def test_common_referee_rejects_mutation_wrong_shape_and_wrong_dtype():
    from kernel_study.correctness import check_forward
    torch = pytest.importorskip('torch')
    x = torch.tensor([2.])
    case = dict(atol=0.,rtol=0.)
    for bad in (lambda v:v.add_(1.), lambda v:(v+1).reshape(1,1), lambda v:(v+1).double()):
        with pytest.raises(AssertionError):
            check_forward(bad, [x.clone()], x+1, case)


@pytest.mark.gpu
def test_layernorm_start_preserves_small_differences_on_large_offsets(tmp_path):
    from kernel_study import public_cases as public
    from kernel_study.correctness import check_forward, fp64_chunks
    from kernel_study.interface import local_forward
    import torch
    case={**public.experiment()['cases']['40'],'output_shape':[3,4,16,79]}
    torch.manual_seed(10001)
    x=torch.rand(case['output_shape'],device='cuda')/32+1024
    model=torch.nn.LayerNorm(x.shape[1:],device='cuda').eval()
    with torch.no_grad():
        w,b=model.weight,model.bias
        source,config=public.start(case)
        path=tmp_path/'device.py'
        path.write_text(source)
        fn=local_forward(case,path,config,[x,w,b])
        actual=check_forward(lambda *_:fn(),[x,w,b],model(x),case,
                                    reference_chunks=fp64_chunks(model,[x]))
        assert actual['elements']==x.numel()


def test_stress_uses_the_fp64_error_of_eager_and_rejects_nonfinite():
    torch=pytest.importorskip('torch')
    from kernel_study.correctness import check_forward
    x = torch.tensor([[1., 1.]])
    expected = torch.tensor([[.9, 1.1]])
    reference = torch.ones_like(expected, dtype=torch.float64)
    case = dict(atol=1.e-6, rtol=0.)
    def check(output):
        return check_forward(lambda *_: output,[x],expected,case,
                             reference_chunks=[(slice(None),reference)])
    assert check(torch.tensor([[1.1, .9]]))['elements']==2
    for invalid in (torch.tensor([[1.2,1.]]), torch.tensor([[float('nan'),1.]])):
        with pytest.raises(AssertionError): check(invalid)


@pytest.mark.gpu
def test_declarative_validation_poisons_each_fresh_output(tmp_path):
    import torch
    from kernel_study import public_cases, interface
    case={**public_cases.experiment()['cases']['26'],'output_shape':[2,17]}
    source,config=public_cases.start(case)
    source=source.replace('i<TOTAL)', 'i<TOTAL-1)')
    path=tmp_path/'partial.py'
    path.write_text(source)
    x=torch.ones((2,17),device='cuda')
    forward,_=interface.prepare(case,config,[x],path,fresh_output=True,poison_output=True)
    for _ in range(2):
        assert torch.isnan(forward().flatten()[-1])
