from kernel_study.core import ROOT
from kernel_study.source_policy import check_source
import pytest


def test_baseline_source_passes_policy():
    check_source((ROOT/'kernel_study/kernels.py').read_text())


@pytest.mark.parametrize('source',[
    'import os\nos.remove("validator.py")',
    'import triton\nx = open("secret")',
    'import triton\n@triton.jit\ndef f(X):\n    eval("1")',
    'import triton\n@triton.jit\ndef f(X):\n    x = X.__class__',
    'import triton\n@triton.autotune([])\ndef f(X):\n    pass',
    'import triton\n@triton.jit\ndef f(X=open("x")):\n    pass',
    'import triton\n@triton.jit\ndef f(X):\n    import os',
])
def test_device_policy_rejects_host_execution(source):
    with pytest.raises(ValueError):
        check_source(source)
