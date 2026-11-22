"""Registered KernelBench tasks, upstream revisions and tensor layouts."""
import importlib.util
import math
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


def layout(case):
    shape = case['output_shape']
    m, n = shape[0], math.prod(shape[1:])
    scalars = {'M':m, 'ROWS':m, 'N':n, 'TOTAL':m*n, 'RB':1 << (n-1).bit_length()}
    names = ['X']
    scratch = 2*m
    if case['family'] in ('matmul','matvec'):
        names, scalars['K'] = ['A','B'], case['k']
    if case['family'] in ('softmax','layernorm'):
        scratch = 2*m*(1 + (n+1023)//1024)
    if case['family'] == 'layernorm':
        names += ['W','B']
    return names,scalars,scratch
