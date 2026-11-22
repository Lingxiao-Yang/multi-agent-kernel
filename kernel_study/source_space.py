"""Source-generation search space with a declarative, trusted launch interface.

No optimized implementation is included in the starting module. The model can
replace kernels and their launch sequence, but cannot execute host Python.
"""
import ast
import copy
import math

from .source_policy import check_source

MODE = 'source-v1'
GATE_VERSION = 3
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
    from .source_policy import ALLOWED_TL
    from .public_cases import layout
    names, scalars, scratch = layout(case)
    return {'format':MODE,'inputs':names, 'scalars':scalars, 'output_shape':case['output_shape'],
            'workspace':{'Y':'output', 'S':[scratch], 'T':case['output_shape']},
            'allowed_tl':sorted(ALLOWED_TL), 'limits':'1-4 declarative launches'}


def _inline_straight_line_temporaries(body):
    """Normalize alias/temporary rewrites for straight-line kernels only.

This constructs a comparison AST, never executed code. Control-flow bodies
retain the conservative structural screen because aliasing across loops/joins
requires a real compiler analysis.
"""
    if any(isinstance(n, (ast.For, ast.If, ast.AugAssign, ast.AnnAssign, ast.NamedExpr))
           for statement in body for n in ast.walk(statement)):
        return body
    if any(isinstance(n,ast.Assign) and (len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name))
           for statement in body for n in ast.walk(statement)):
        return body
    bindings, result = {}, []
    class Resolve(ast.NodeTransformer):
        def visit_Name(self,node):
            return copy.deepcopy(bindings[node.id]) if isinstance(node.ctx,ast.Load) and node.id in bindings else node
    for statement in body:
        if isinstance(statement,ast.Assign):
            value=Resolve().visit(copy.deepcopy(statement.value))
            if sum(1 for _ in ast.walk(value)) > 10000:
                return body
            bindings[statement.targets[0].id]=value
        elif not isinstance(statement,ast.Pass):
            value=Resolve().visit(copy.deepcopy(statement))
            if sum(1 for _ in ast.walk(value)) > 10000:
                return body
            result.append(value)
    return result


def _body_shapes(source, config):
    """Conservative structural screen, not a proof of semantic improvement.

Ignore identifiers, literals and docstrings to reject trivial rewrites; inspect
only launched functions. Human review of retained diffs remains necessary.
"""
    tree = check_source(source)
    funcs = {x.name:x for x in tree.body if isinstance(x, ast.FunctionDef)}
    result = []
    class Normalize(ast.NodeTransformer):
        def visit_BinOp(self,node):
            node=self.generic_visit(node)
            if not isinstance(node.op,(ast.Add,ast.Mult,ast.BitAnd,ast.BitOr)):
                return node
            terms=[]
            def flatten(part):
                if isinstance(part,ast.BinOp) and type(part.op) is type(node.op):
                    flatten(part.left); flatten(part.right)
                else: terms.append(part)
            flatten(node)
            result=terms[0]
            for term in terms[1:]: result=ast.BinOp(left=result,op=node.op,right=term)
            return result
        def visit_Call(self,node):
            node=self.generic_visit(node)
            signatures={'tl.load':['pointer','mask','other'], 'tl.store':['pointer','value','mask'],
                        'tl.sum':['input','axis'], 'tl.max':['input','axis']}
            signature=signatures.get(ast.unparse(node.func),[])
            remaining={k.arg:k.value for k in node.keywords}
            for index,name in enumerate(signature):
                if index==len(node.args) and name in remaining:
                    node.args.append(remaining.pop(name))
            node.keywords=[ast.keyword(arg=k,value=v) for k,v in sorted(remaining.items(),key=lambda x:str(x[0]))]
            return node
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
        body = _inline_straight_line_temporaries(body)
        result.append(ast.dump(Normalize().visit(ast.Module(body=copy.deepcopy(body), type_ignores=[])), include_attributes=False))
    return result


def check_edit(source, config, parent_source, parent_config):
    validate_plan(config)
    before = _body_shapes(parent_source, parent_config)
    after = _body_shapes(source, config)
    # Dropping a launch alone is not device-code authorship either.
    if all(body in before for body in after):
        raise ValueError('source_edit_required: launched device bodies have no structural change beyond identifiers/literals/straight-line temporaries')
    return {'structural_edit':True, 'launches_before':len(before), 'launches_after':len(after),
            'note':'Structural screen passed; transformation mechanism requires source review.'}


def prepare(case, config, inputs, source, *, fresh_output=False, poison_output=False):
    import torch
    from .source_policy import load_source
    validate_plan(config)
    _body_shapes(open(source, encoding='utf-8').read(), config)
    mod = load_source(source)
    from .public_cases import layout
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
