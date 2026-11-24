import os
import shutil
import subprocess
import sys
import tarfile
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


def test_slurm_profile_extracts_with_the_cluster_tar(tmp_path):
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path/'kernel-study-campaign/a100-matvec-20261122'
    work, node, tools = tmp_path/'work', tmp_path/'node', tmp_path/'bin'
    for directory in (archive/'venv/bin', work, node, tools):
        directory.mkdir(parents=True)
    (archive/'cuda-module').write_text('cuda/test')
    interpreter = archive/'venv/bin/python'
    interpreter.write_text('#!/bin/sh\ncase "$0" in "$SLURM_TMPDIR"/*) echo local;; *) echo shared;; esac\n')
    interpreter.chmod(0o700)
    with tarfile.open(work/'runtime.tar.zst','w') as bundle:
        bundle.add(archive/'venv', arcname='venv')
    (work/'runtime.bytes').write_text('10000\n')
    tar = shutil.which('tar')
    for name, body in [('tar', f'for arg do [ "$arg" != "--zstd" ] || exit 64; done\nexec "{tar}" "$@"'),
                       ('zstd', 'cat'), ('findmnt', 'echo xfs')]:
        path = tools/name
        path.write_text('#!/bin/sh\n'+body+'\n')
        path.chmod(0o700)
    env = dict(os.environ, PATH=str(tools)+':'+os.environ['PATH'], SCRATCH=str(tmp_path),
               SLURM_TMPDIR=str(node), SLURM_JOB_ID='123', SLURM_SUBMIT_DIR=str(work))
    result = subprocess.run(['bash','-c','module(){ :; }; export -f module; bash "$1"','bash',
                             str(root/'scripts/ls6_verifier_profile.sbatch')], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['shared','local']
    assert not any(node.iterdir())
