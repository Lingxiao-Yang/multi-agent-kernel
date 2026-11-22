"""Check numerical outputs and unchanged inputs against eager and FP64 references."""
import copy


def stress_input(x, edge):
    if edge == 'zeros':
        x.zero_()
    elif edge == 'constant':
        x.fill_(3.)
    elif edge == 'large':
        x.mul_(20.)
    elif edge == 'negative':
        x.neg_().sub_(1.)
    elif edge == 'large_offset':
        x.mul_(1/32).add_(1024.)
    else:
        raise ValueError(edge)


def fp64_chunks(model, inputs):
    model64 = copy.deepcopy(model).double()
    first, *rest = inputs
    other = [v.double() for v in rest]
    rows = max(1, 2**20 // first[0].numel())
    for i in range(0, first.shape[0], rows):
        part = slice(i, i+rows)
        yield part, model64(first[part].double(), *other)


def check_forward(forward, inputs, expected, case, *, reference_chunks=None):
    import torch
    snapshots = [x.clone() for x in inputs]
    actual = forward(*inputs)
    if not all(torch.equal(x,s) for x,s in zip(inputs,snapshots)):
        raise AssertionError('input or parameter was mutated')
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError('output metadata differs from reference')
    delta = torch.zeros((),device=actual.device)
    bad = torch.zeros((),dtype=torch.int64,device=actual.device)
    reference_info = {'reference':'eager'}
    if reference_chunks is None:
        for a,b in zip(actual.flatten().split(2**23), expected.flatten().split(2**23)):
            a,b = a.float(),b.float()
            delta = torch.maximum(delta,(a-b).abs().max())
            bad += (~torch.isclose(a,b,atol=case['atol'],rtol=case['rtol'],equal_nan=False)).sum()
    else:
        eager_error = torch.zeros((),dtype=torch.float64,device=actual.device)
        for part,reference in reference_chunks:
            a,b = actual[part].double(), expected[part].double()
            error = (a-reference).abs()
            eager_delta = (b-reference).abs()
            allowed = eager_delta + case['atol'] + case['rtol']*reference.abs()
            delta = torch.maximum(delta,error.max())
            eager_error = torch.maximum(eager_error,eager_delta.max())
            bad += (~(torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(reference) & (error <= allowed))).sum()
        reference_info = {'reference':'fp64','eager_max_abs_error':float(eager_error)}
    count = int(bad)
    if count:
        raise AssertionError(f'{count}/{actual.numel()} elements outside tolerance, max_abs_error={float(delta)}')
    return {'elements':actual.numel(),'max_abs_error':float(delta),'inputs_unchanged':True,**reference_info}
