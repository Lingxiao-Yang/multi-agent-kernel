"""Check that a proposed edit changes the launched device code."""
import ast
import copy

from .interface import launched_functions, validate_plan

GATE_VERSION = 3
CONTRACT = '''For implement return {"config":{"mode":"source-v1","launches":[{"kernel":"name","grid":[positive integers],"args":[buffer/scalar symbol names or integer constants],"num_warps":4,"num_stages":1}]},"source":"complete Triton module", "rationale":"code transformation and hypothesis"}.
Write new device code: config-only, null source, cosmetic renaming, and constant-only source edits are rejected and consume an attempt. Change computation, data movement, parallel decomposition, or kernel structure. No optimized alternative is supplied. You may replace function names/signatures and use 1-4 launches, choosing arbitrary bounded positive grids and integer constexpr arguments. Supply the complete source on every attempt, preserving any kernels still launched.
Only imports triton and triton.language as tl, and @triton.jit device functions are allowed. Inline all device logic; no helper function calls, host code, autotune, reference/timer edits, input mutation, or external access. Supported primitives are listed in launch_contract. Every output must be overwritten on each invocation; scratch is uninitialized and must be written before being read. Shapes are contiguous row-major. Other stages return concise JSON; do not reproduce full source outside implement. Review the actual proposed source and measured failures.'''


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
    for fn in launched_functions(source, config):
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
