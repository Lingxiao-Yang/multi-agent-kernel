"""Frozen KernelBench inputs and shared Triton starts for both agent interfaces."""
import ast
import importlib.util
import math
import os
import subprocess
from pathlib import Path

from .core import ROOT, read_json
from . import source_space


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
    legacy = ROOT/'kernel_study/kernels.py'
    public = ROOT/'kernel_study/public_kernels.py'
    kernels, launches = [], []
    def add(path, name, grid, args, stages=1):
        source = device_function(path,name)
        if name == 'matmul_tensor':
            source = source.replace('tl.dot(a, b, acc)', "tl.dot(a, b, acc, input_precision='ieee')")
        kernels.append(source)
        launches.append(dict(kernel=name,grid=grid,args=args,num_warps=4,num_stages=stages))
    if family == 'matvec':
        add(matvec_path(),'_matvec_kernel',[(m+127)//128],
            ['A','B','Y','M','K','K',1,1,1,1,1,128,256],2)
    elif family == 'matmul':
        add(legacy,'matmul_tensor',[(m+31)//32,(n+31)//32],['A','B','Y','M','N','K',32,32,32])
    elif family in ('softmax','layernorm'):
        groups = (n+1023)//1024
        flag = int(family=='softmax')
        add(public,'row_partials',[groups,m],['X','S','ROWS','N',groups,1024,flag])
        add(public,'row_combine',[m],['S','ROWS','N',groups,1024,1 << (groups-1).bit_length(),flag])
        operands = ['X'] if flag else ['X','W','B']
        add(legacy,family+'_apply',[(total+2047)//2048],operands+['S','Y','TOTAL','N',2048])
    elif family == 'gelu':
        add(public,'gelu',[(total+2047)//2048],['X','Y','TOTAL',2048])
    elif family == 'rmsnorm':
        spatial = math.prod(shape[2:])
        add(public,'rms_norm',[(spatial+127)//128,m],['X','Y',shape[1],spatial,128])
    else:
        raise ValueError(family)
    return 'import triton\nimport triton.language as tl\n\n' + '\n\n'.join(kernels)+'\n', {
        'mode':source_space.MODE,'launches':launches}


def render_native(case, source, config):
    source_space.validate_plan(config)
    source_space._body_shapes(source,config)
    names, scalars, scratch = layout(case)
    arguments = {name:name for name in names}
    if case['family']=='layernorm':
        arguments.update(W='weight', B='bias')
    signature = ', '.join(arguments[n] for n in names)
    lines = [source,'import torch',f'def kernel_function({signature}):',
             f"    Y = {arguments[names[0]]}.new_empty({tuple(case['output_shape'])!r})"]
    args = {a for item in config['launches'] for a in item['args']}
    if 'S' in args: lines.append(f'    S = Y.new_empty(({scratch},), dtype=torch.float32)')
    if 'T' in args: lines.append('    T = torch.empty_like(Y)')
    for item in config['launches']:
        values = [str(scalars.get(a,arguments.get(a,a))) for a in item['args']]
        values += [f'{k}={item[k]}' for k in ('num_warps','num_stages')]
        lines.append(f"    {item['kernel']}[{tuple(item['grid'])!r}]({', '.join(values)})")
    return '\n'.join([*lines,'    return Y',''])


def local_forward(case, path, config, inputs):
    return source_space.prepare(case,config,inputs,path,fresh_output=True)[0]


def check_forward(forward, inputs, expected, case):
    import torch
    snapshots = [x.clone() for x in inputs]
    actual = forward(*inputs)
    if not all(torch.equal(x,s) for x,s in zip(inputs,snapshots)):
        raise AssertionError('input or parameter was mutated')
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError('output metadata differs from reference')
    delta = torch.zeros((),device=actual.device)
    bad = torch.zeros((),dtype=torch.int64,device=actual.device)
    for a,b in zip(actual.flatten().split(2**23),expected.flatten().split(2**23)):
        a,b = a.float(),b.float()
        diff = (a-b).abs()
        delta = torch.maximum(delta,diff.max())
        bad += (~torch.isclose(a,b,atol=case['atol'],rtol=case['rtol'],equal_nan=False)).sum()
    count = int(bad)
    if count:
        raise AssertionError(f'{count}/{actual.numel()} elements outside tolerance, max_abs_error={float(delta)}')
    return {'elements':actual.numel(),'max_abs_error':float(delta),'inputs_unchanged':True}


def evaluate(request):
    import time
    import torch
    from . import bench
    from .core import environment
    case, protocol = request['case'], request['protocol']
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    def bind(source, config, interface, buffers):
        if interface == 'native':
            native = load_native(source)
            return lambda: native(*buffers), []
        return source_space.prepare(case,config,buffers,source,fresh_output=True)
    source = request['source']
    config = request.get('config')
    interface = request.get('interface','declarative')
    seeds = protocol['final_validation_seeds' if request.get('final') else 'validation_seeds']
    with torch.no_grad():
        records = []
        for seed in seeds:
            model,inputs,buffers = task_inputs(case,seed)
            expected = model(*inputs)
            forward,_ = bind(source,config,interface,buffers)
            first = time.perf_counter()
            record = check_forward(lambda *args: forward(),buffers,expected,case)
            if not records: first_check_seconds = time.perf_counter()-first
            records.append(dict(seed=seed,edge='random',**record))
            del model,inputs,buffers,expected,forward
        validation_seconds = time.perf_counter()-started
        model,inputs,buffers = task_inputs(case,seeds[0]+77)
        forward,compiled = bind(source,config,interface,buffers)
        parent,_ = bind(request.get('incumbent_source',source),request.get('incumbent_config',config),
                        request.get('incumbent_interface',interface),buffers)
        candidate,incumbent = bench.time_pair(forward,parent,protocol)
        speedup = incumbent['median_ms']/candidate['median_ms']
        result = dict(status='ok',environment=environment(),validation=records,validation_seconds=validation_seconds,
                      first_validation_seconds=first_check_seconds,candidate=candidate,incumbent=incumbent,
                      paired_speedup=speedup,confirmed=False)
        if speedup > protocol['improvement_threshold']:
            c,i = bench.time_pair(forward,parent,protocol)
            result['confirmation'] = {'candidate':c,'incumbent':i}
            result['confirmed'] = i['median_ms']/c['median_ms'] > protocol['improvement_threshold']
        if request.get('library'):
            result['library_candidate'],result['library'] = bench.time_pair(forward,lambda: model(*inputs),protocol)
        if request.get('final'):
            compiled_model = torch.compile(model,mode=experiment()['compile_mode'],fullgraph=True)
            result['compile_candidate'],result['compile'] = bench.time_pair(forward,lambda: compiled_model(*inputs),protocol)
        if request.get('profile'):
            result['profile'] = {'mode':'compiler-resources','kernels':bench.resource_summary(compiled)}
    result['worker_seconds'] = time.perf_counter()-started
    return result
