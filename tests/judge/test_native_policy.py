import pytest

from kernel_study.judge.policy import check_source
from kernel_study.tasks.kernelbench import experiment
from kernel_study.tasks.starts import start


@pytest.mark.parametrize('case', list(experiment()['cases'].values()))
def test_registered_native_programs_pass(case):
    check_source(start(case))


BASE = '''import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(x, out, n: tl.constexpr):
    i = tl.arange(0, n)
    tl.store(out + i, tl.load(x + i))

def kernel_function(x):
    out = torch.empty_like(x)
    copy_kernel[(1,)](x, out, 128)
    return out
'''


@pytest.mark.parametrize('source', [
    BASE.replace('return out', 'return torch.sin(x)'),
    BASE.replace('return out', 'return x + out'),
    BASE.replace('def kernel_function(x):', 'cache = {}\ndef kernel_function(x):').replace('return out', 'cache[0] = out\n    return out'),
    BASE.replace('return out', 'x.copy_(out)\n    return out'),
    BASE.replace('out + i, tl.load(x + i)', 'x + i, tl.load(out + i)'),
    BASE.replace('return out', 'return x'),
])
def test_both_agents_reject_library_results_cached_state_or_writes_to_inputs(source):
    with pytest.raises(ValueError):
        check_source(source)


def test_accepts_launch_tuning_and_a_candidate_owned_workspace():
    source=BASE.replace('copy_kernel[(1,)](x, out, 128)', '''partial = torch.empty((128,), device=x.device, dtype=torch.float32)
    copy_kernel[(1,)](x, partial, 128, num_warps=2)
    copy_kernel[(1,)](partial, out, 128)''')
    check_source(source)


def test_wrapper_must_launch_a_kernel_on_every_return_path():
    launch = '    copy_kernel[(1,)](x, out, 128)'
    for source in (BASE.replace(launch, ''), BASE.replace(launch, '    if x.is_cuda:\n    '+launch)):
        with pytest.raises(ValueError, match='launch'):
            check_source(source)
    check_source(BASE.replace(launch, '    if not x.is_cuda:\n        raise ValueError("expected CUDA")\n'+launch))


def test_module_functions_cannot_be_overwritten_by_constants():
    with pytest.raises(ValueError):
        check_source(BASE + '\ncopy_kernel = 1\n')


def test_allocation_must_not_replace_triton_computation():
    with pytest.raises(ValueError):
        check_source(BASE.replace('return out', 'return torch.zeros_like(x)'))


def test_device_tensor_methods_do_not_require_a_primitive_whitelist():
    check_source(BASE.replace('tl.load(x + i)', 'tl.load(x + i).sum(0)'))
