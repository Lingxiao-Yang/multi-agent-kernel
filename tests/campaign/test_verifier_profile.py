import subprocess
import sys
from pathlib import Path

from scripts.profile_start_verifier import stage_times


def test_timing_counts_exclusive_stages_and_keeps_the_last_stage_on_timeout(tmp_path):
    trace = tmp_path/'stages.tsv'
    trace.write_text('11 imports\n13 process\n14 inputs\n15 version\n17 inputs\n')
    times, last = stage_times(trace, 10, 20)
    assert times == {'process':2, 'imports':2, 'inputs':4, 'version':2}
    assert last == 'inputs'
    assert stage_times(tmp_path/'absent', 10, 20) == ({'process':10}, 'process')


def test_probe_runs_the_original_check_and_times_its_operations(tmp_path):
    root = Path(__file__).resolve().parents[2]
    (tmp_path/'test_kernel.py').write_bytes((root/'scripts/trace_verifier_stages.py').read_bytes())
    (tmp_path/'original_test.py').write_text("""
from types import SimpleNamespace
torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
def task_inputs(*args):
    kernelbench.upstream('KernelBench')
    kernelbench.load_module(None)
    return (lambda x: x+1), [2], [2]
kernelbench = SimpleNamespace(task_inputs=task_inputs, upstream=lambda name: None, load_module=lambda path: None)
def kernel_function(x): return x+1
def check_forward(fn, buffers, expected, case): assert fn(*buffers) == expected == 3
def test_kernel():
    model, inputs, buffers = kernelbench.task_inputs({}, 101)
    check_forward(kernel_function, buffers, model(*inputs), {})
""")
    result = subprocess.run([sys.executable, str(tmp_path/'test_kernel.py')], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == 'PASS'
    stages = [line.split()[1] for line in (tmp_path/'stages.tsv').read_text().splitlines()]
    assert set(stages) == {'imports', 'version', 'inputs', 'reference', 'kernel', 'comparison', 'process'}
