"""Frozen KernelBench inputs and shared Triton starts for both agent interfaces."""
import ast
import importlib.util
import math
import os
import subprocess
from pathlib import Path

from .core import ROOT, read_json

MODE = 'source-v1'


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


def load_native(path):
    return load_module(path).kernel_function


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


def device_function(path,name):
    text = Path(path).read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '@triton.jit\n' + ast.get_source_segment(text,fn)


def start(case):
    family = case['family']
    shape = case['output_shape']
    _, sizes, _ = layout(case)
    m,n,total = (sizes[k] for k in ('M','N','TOTAL'))
    public = ROOT/'kernel_study/public_kernels.py'
    kernels, launches = [], []
    def add(path, name, grid, args, stages=1):
        source = device_function(path,name)
        kernels.append(source)
        launches.append(dict(kernel=name,grid=grid,args=args,num_warps=4,num_stages=stages))
    if family == 'matvec':
        add(matvec_path(),'_matvec_kernel',[(m+127)//128],
            ['A','B','Y','M','K','K',1,1,1,1,1,128,256],2)
    elif family == 'matmul':
        add(public,'matmul_tensor',[(m+31)//32,(n+31)//32],['A','B','Y','M','N','K',32,32,32])
    elif family in ('softmax','layernorm'):
        groups = (n+1023)//1024
        flag = int(family=='softmax')
        if flag:
            add(public,'row_partials',[groups,m],['X','S','ROWS','N',groups,1024,flag])
        else:
            add(public,'layernorm_partials',[groups,m],['X','S','ROWS','N',groups,1024])
        add(public,'row_combine',[m],['S','ROWS','N',groups,1024,1 << (groups-1).bit_length(),flag])
        operands = ['X'] if flag else ['X','W','B']
        add(public,family+'_apply',[(total+2047)//2048],operands+['S','Y','TOTAL','N',2048])
    elif family == 'gelu':
        add(public,'gelu',[(total+2047)//2048],['X','Y','TOTAL',2048])
    elif family == 'rmsnorm':
        spatial = math.prod(shape[2:])
        add(public,'rms_norm',[(spatial+127)//128,m],['X','Y',shape[1],spatial,128])
    else:
        raise ValueError(family)
    return 'import triton\nimport triton.language as tl\n\n' + '\n\n'.join(kernels)+'\n', {
        'mode':MODE,'launches':launches}
