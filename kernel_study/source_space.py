"""Source-generation search space with a declarative, trusted launch interface.

No optimized implementation is included in the starting module. The model can
replace kernels and their launch sequence, but cannot execute host Python.
"""
import ast
import copy
import math

from .core import ROOT
from .source_policy import check_source

MODE = 'source-v1'
SYMBOLS = {'X', 'A', 'B', 'W', 'Y', 'S', 'T', 'M', 'N', 'K', 'ROWS', 'TOTAL', 'RB'}
CONTRACT = '''For implement return {"config":{"mode":"source-v1","launches":[{"kernel":"name","grid":[positive integers],"args":[buffer/scalar symbol names or integer constants],"num_warps":4,"num_stages":1}]},"source":"complete Triton module", "rationale":"code transformation and hypothesis"}.
Write new device code: config-only, null source, cosmetic renaming, and constant-only source edits are rejected and consume an attempt. Change computation, data movement, parallel decomposition, or kernel structure. No optimized alternative is supplied. You may replace function names/signatures and use 1-4 launches, choosing arbitrary bounded positive grids and integer constexpr arguments. Supply the complete source on every attempt, preserving any kernels still launched.
Only imports triton and triton.language as tl, and @triton.jit device functions are allowed. Inline all device logic; no helper function calls, host code, autotune, reference/timer edits, input mutation, or external access. Supported primitives are listed in launch_contract. Every output must be overwritten on each invocation; scratch is uninitialized and must be written before being read. Shapes are contiguous row-major. Other stages return concise JSON; do not reproduce full source outside implement. Review the actual proposed source and measured failures.'''


def validate_plan(config):
    if set(config) != {'mode', 'launches'} or config['mode'] != MODE:
        raise ValueError('source config requires mode=source-v1 and launches only')
    launches = config['launches']
    if not isinstance(launches, list) or not 1 <= len(launches) <= 4:
        raise ValueError('require 1-4 launches')
    for item in launches:
        if set(item) != {'kernel', 'grid', 'args', 'num_warps', 'num_stages'}:
            raise ValueError('launch requires kernel/grid/args/num_warps/num_stages')
        if not isinstance(item['kernel'], str) or not item['kernel'].isidentifier() or item['kernel'].startswith('_'):
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


def baseline(case):
    f = case['family']
    rows, n = case['shape'][:2]
    def launch(name, grid, args):
        return dict(kernel=name, grid=grid, args=args, num_warps=4, num_stages=1)
    flat = [(rows*n + 255)//256]
    if f == 'matmul':
        launches = [launch('matmul_simt', [(rows+15)//16, (n+15)//16], ['A','B','Y','M','N','K',16,16,32])]
    elif f in ['softmax', 'layernorm']:
        inputs = ['X'] if f == 'softmax' else ['X','W','B']
        launches = [launch(f+'_stats', [rows], ['X','S','N','RB']),
                    launch(f+'_apply', flat, inputs+['S','Y','TOTAL','N',256])]
    else:
        launches = [launch('bias_add', flat, ['X','B','T','TOTAL','N',256]),
                    launch('relu', flat, ['T','Y','TOTAL',256])]
    names = {x['kernel'] for x in launches}
    tree = ast.parse((ROOT/'kernel_study/kernels.py').read_text(encoding='utf-8'))
    functions = [ast.unparse(x) for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
    source = 'import triton\nimport triton.language as tl\n\n' + '\n\n'.join(functions) + '\n'
    return source, {'mode':MODE, 'launches':launches}


def contract(case):
    from .source_policy import ALLOWED_TL
    f = case['family']
    return {'format':MODE, 'buffers':({'A':'[M,K] input', 'B':'[K,N] input'} if f == 'matmul' else
            {'X':'[ROWS,N] input', **({'W':'[N] input weight', 'B':'[N] input bias'} if f == 'layernorm' else {'B':'[N] input bias'} if f == 'elementwise' else {})}),
            'workspace':{'Y':'output, same dtype as inputs', 'S':'[ROWS,2] float32 scratch', 'T':'output-shaped scratch, same dtype as output'},
            'scalars':'M/ROWS=shape[0], N=shape[1], K=shape[2] for matmul, TOTAL=M*N, RB=next_power_of_2(N)',
            'grid':'Concrete integer grid dimensions for this case, no expressions or Python. args are positional, symbols resolve to trusted buffers/scalars; integer literals become constexpr parameters.',
            'allowed_tl':sorted(ALLOWED_TL), 'limits':'1-4 launches; num_warps 1/2/4/8; num_stages 1-4; layernorm epsilon is 1e-5'}


def _body_shapes(source, config):
    """Conservative structural screen, not a proof of semantic improvement.

Ignore identifiers, literals and docstrings to reject trivial rewrites; inspect
only launched functions. Human review of retained diffs remains necessary.
"""
    tree = check_source(source)
    funcs = {x.name:x for x in tree.body if isinstance(x, ast.FunctionDef)}
    result = []
    class Normalize(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.Name(id=node.id if node.id in ['tl','triton','range','float','int','min','max'] else 'var', ctx=node.ctx)
        def visit_Constant(self, node):
            return ast.Constant(value='<literal>')
    for item in config['launches']:
        if item['kernel'] not in funcs:
            raise ValueError('launch references missing kernel: '+item['kernel'])
        fn = funcs[item['kernel']]
        if len(fn.args.args) != len(item['args']):
            raise ValueError('launch argument count does not match kernel signature')
        body = [x for x in fn.body if not (isinstance(x, ast.Expr) and isinstance(x.value, ast.Constant) and isinstance(x.value.value, str))]
        result.append(ast.dump(Normalize().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[])), include_attributes=False))
    return result


def check_edit(source, config, parent_source, parent_config):
    validate_plan(config)
    before = _body_shapes(parent_source, parent_config)
    after = _body_shapes(source, config)
    # Dropping a launch alone is not device-code authorship either.
    if all(body in before for body in after):
        raise ValueError('source_edit_required: launched device bodies have no structural change beyond identifiers/literals')
    return {'structural_edit':True, 'launches_before':len(before), 'launches_after':len(after),
            'note':'Structural screen passed; transformation mechanism requires source review.'}


def prepare(case, config, inputs, source):
    import torch
    from .source_policy import load_source
    validate_plan(config)
    _body_shapes(open(source, encoding='utf-8').read(), config)
    mod = load_source(source)
    m,n = case['shape'][:2]
    output = torch.empty((m,n), dtype=inputs[0].dtype, device='cuda')
    buffers = {'Y':output, 'S':torch.empty((m,2),device='cuda'), 'T':torch.empty_like(output),
               'M':m, 'ROWS':m, 'N':n, 'TOTAL':m*n, 'RB':1 << (n-1).bit_length()}
    if case['family'] == 'matmul':
        buffers.update(A=inputs[0],B=inputs[1],K=case['shape'][2])
    else:
        buffers['X'] = inputs[0]
        if case['family'] == 'layernorm': buffers.update(W=inputs[1],B=inputs[2])
        elif case['family'] == 'elementwise': buffers['B'] = inputs[1]
    launches = []
    for item in config['launches']:
        args = [buffers[a] if isinstance(a,str) else a for a in item['args']]
        launches.append((getattr(mod,item['kernel']),tuple(item['grid']),args,
                         dict(num_warps=item['num_warps'],num_stages=item['num_stages'])))
    compiled = []
    def launch():
        for kernel,grid,args,opts in launches:
            built = kernel[grid](*args,**opts)
            if all(built is not old for old in compiled): compiled.append(built)
        return output
    return launch, compiled
