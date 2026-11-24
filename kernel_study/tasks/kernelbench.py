"""Registered KernelBench tasks, upstream revisions and tensor layouts."""
import importlib.util
import os
import subprocess
from pathlib import Path

from ..records import ROOT, read_json


def experiment():
    return read_json(ROOT/'configs/agent_baseline.json')


def upstream(name):
    path = Path(os.environ['KERNEL_BASELINE_UPSTREAM']) / name
    head = subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
    if head != experiment()['upstream'][name]:
        raise ValueError(f'{name} revision {head} differs from the registered revision')
    subprocess.run(['git','-C',str(path),'diff','--quiet','HEAD'],check=True)
    return path


def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_path(case):
    return next((upstream('KernelBench')/'KernelBench/level1').glob(f"{case['public_id']}_*.py"))


def matvec_path():
    return upstream('KernelAgent')/'examples/optimize_01_matvec/input.py'


def task_inputs(case, seed):
    import torch
    task = load_module(task_path(case))
    torch.manual_seed(seed)
    dtype = getattr(torch,case['dtype'])
    with torch.device('cuda'):
        model = task.Model(*task.get_init_inputs()).to(dtype).eval()
        inputs = [x.to(dtype) for x in task.get_inputs()]
    parameters = [model.ln.weight, model.ln.bias] if case['family']=='layernorm' else []
    return model, inputs, inputs+parameters


def test_source(case):
    return f"""from pathlib import Path
from kernel_study.judge.policy import check_source

check_source(Path(__file__).with_name('kernel.py').read_text())

import torch
from kernel import kernel_function
from kernel_study.tasks import kernelbench
from kernel_study.judge.correctness import check_forward

def test_kernel():
    plan = kernelbench.experiment()
    case = plan['cases'][{str(case['public_id'])!r}]
    with torch.no_grad():
        model, inputs, buffers = kernelbench.task_inputs(case, plan['validation_seeds'][0])
        check_forward(kernel_function, buffers, model(*inputs), case)
    return True

if __name__ == '__main__':
    test_kernel()
    print('PASS')
"""
