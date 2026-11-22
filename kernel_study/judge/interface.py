"""Validate and execute declarative launches and render native wrappers."""
import ast
import math
from pathlib import Path

from ..tasks.kernelbench import layout
from ..tasks.starts import MODE
from .policy import ALLOWED_TL, check_source, load_source

SYMBOLS = {'X', 'A', 'B', 'W', 'Y', 'S', 'T', 'M', 'N', 'K', 'ROWS', 'TOTAL', 'RB'}


def validate_plan(config):
    if set(config) != {'mode', 'launches'} or config['mode'] != MODE:
        raise ValueError('source config requires mode=source-v1 and launches only')
    launches = config['launches']
    if not isinstance(launches, list) or not 1 <= len(launches) <= 4:
        raise ValueError('require 1-4 launches')
    for item in launches:
        if set(item) != {'kernel', 'grid', 'args', 'num_warps', 'num_stages'}:
            raise ValueError('launch requires kernel/grid/args/num_warps/num_stages')
        if not isinstance(item['kernel'], str) or not item['kernel'].isidentifier() or item['kernel'].startswith('__'):
            raise ValueError('invalid kernel identifier')
        grid = item['grid']
        if not isinstance(grid, list) or not 1 <= len(grid) <= 3 or any(type(x) is not int or not 1 <= x <= 2**20 for x in grid) or math.prod(grid) > 2**22:
            raise ValueError('invalid or oversized launch grid')
        if type(item['num_warps']) is not int or item['num_warps'] not in [1, 2, 4, 8] or type(item['num_stages']) is not int or not 1 <= item['num_stages'] <= 4:
            raise ValueError('unsupported compiler options')
        if not isinstance(item['args'], list) or not 1 <= len(item['args']) <= 32:
            raise ValueError('require 1-32 positional arguments')
        for arg in item['args']:
            if not ((type(arg) is int and 0 <= arg <= 2**24) or (isinstance(arg, str) and arg in SYMBOLS)):
                raise ValueError('invalid argument symbol or integer')
    return config


def contract(case):
    names, scalars, scratch = layout(case)
    return {'format':MODE,'inputs':names, 'scalars':scalars, 'output_shape':case['output_shape'],
            'workspace':{'Y':'output', 'S':[scratch], 'T':case['output_shape']},
            'allowed_tl':sorted(ALLOWED_TL), 'limits':'1-4 declarative launches'}


def launched_functions(source, config):
    tree = check_source(source)
    funcs = {x.name:x for x in tree.body if isinstance(x, ast.FunctionDef)}
    launched = []
    for item in config['launches']:
        if item['kernel'] not in funcs:
            raise ValueError('launch references missing kernel: '+item['kernel'])
        fn = funcs[item['kernel']]
        if len(fn.args.args) != len(item['args']):
            raise ValueError('launch argument count does not match kernel signature')
        launched.append(fn)
    return launched


def prepare(case, config, inputs, source, *, fresh_output=False, poison_output=False):
    import torch
    validate_plan(config)
    launched_functions(Path(source).read_text(encoding='utf-8'), config)
    mod = load_source(source)
    names, scalars, scratch = layout(case)
    buffers = {**scalars, **dict(zip(names, inputs, strict=True))}
    out_shape, scratch_shape = case['output_shape'], (scratch,)
    launches = []
    work_names = {'Y','S','T'}
    used = {a for item in config['launches'] for a in item['args'] if a in work_names}
    def workspace():
        output = inputs[0].new_empty(out_shape)
        work = {'Y':output}
        if 'S' in used: work['S'] = output.new_empty(scratch_shape, dtype=torch.float32)
        if 'T' in used: work['T'] = torch.empty_like(output)
        return work
    cached = None if fresh_output else workspace()
    for item in config['launches']:
        args = [buffers.get(a, a) for a in item['args']]
        slots = [(i, a) for i,a in enumerate(item['args']) if a in work_names]
        launches.append((getattr(mod,item['kernel'])[tuple(item['grid'])],args,slots,
                         dict(num_warps=item['num_warps'],num_stages=item['num_stages'])))
    compiled = []
    def launch():
        work = workspace() if fresh_output else cached
        if poison_output:
            work['Y'].fill_(float('nan'))
        for kernel,template,slots,opts in launches:
            args = template.copy()
            for i, name in slots: args[i] = work[name]
            built = kernel(*args,**opts)
            if all(built is not old for old in compiled): compiled.append(built)
        return work['Y']
    return launch, compiled


def render_native(case, source, config):
    validate_plan(config)
    launched_functions(source,config)
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
    return prepare(case,config,inputs,path,fresh_output=True)[0]
