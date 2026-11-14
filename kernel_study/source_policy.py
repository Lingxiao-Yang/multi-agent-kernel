"""Reject host Python in model-authored device code before importing it.

This is an integrity gate, NOT an adversarial security sandbox. Device code
can still contain out-of-bounds accesses; run workers in an isolated GPU job.
"""
import ast
import importlib.util
from pathlib import Path

ALLOWED_TL = set('constexpr float16 float32 int32 int64 program_id arange full zeros load store max min sum exp log sqrt rsqrt abs maximum minimum where dot trans reshape broadcast_to multiple_of max_contiguous cdiv div_rn range static_range num_programs'.split())


def check_source(source):
    tree = ast.parse(source)
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                if (alias.name, alias.asname) not in [('triton', None), ('triton.language', 'tl')]:
                    raise ValueError('only imports of triton and triton.language as tl are allowed')
            continue
        if not isinstance(statement, ast.FunctionDef):
            raise ValueError('module may contain only imports and JIT functions')
        if len(statement.decorator_list) != 1 or ast.unparse(statement.decorator_list[0]) != 'triton.jit':
            raise ValueError('every candidate function must use @triton.jit only')
        if statement.args.defaults or statement.args.kw_defaults:
            raise ValueError('function defaults are forbidden')
        if statement.returns or statement.args.vararg or statement.args.kwarg or statement.args.kwonlyargs or statement.args.posonlyargs:
            raise ValueError('only plain kernel argument signatures allowed')
        for arg in statement.args.args:
            if arg.annotation is not None and ast.unparse(arg.annotation) != 'tl.constexpr':
                raise ValueError('only tl.constexpr annotations allowed')
        for node in ast.walk(statement):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.Lambda,
                                 ast.With, ast.Try, ast.While, ast.Delete, ast.Raise, ast.Yield)):
                raise ValueError('forbidden construct: ' + type(node).__name__)
            if isinstance(node, ast.Name) and node.id.startswith('__'):
                raise ValueError('dunder name forbidden')
            if isinstance(node, ast.Attribute):
                if node.attr.startswith('_'):
                    raise ValueError('private attribute forbidden')
                if isinstance(node.value, ast.Name) and node.value.id == 'tl':
                    if node.attr not in ALLOWED_TL:
                        raise ValueError('unsupported Triton primitive ' + node.attr)
                elif isinstance(node.value, ast.Name) and node.value.id == 'triton':
                    if node.attr not in ['jit', 'cdiv']:
                        raise ValueError('unsupported triton attribute')
                elif node.attr != 'to':
                    raise ValueError('only tensor .to access allowed')
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id not in ['range', 'float', 'int', 'min', 'max']:
                    raise ValueError('arbitrary function calls forbidden')
                if not isinstance(node.func, (ast.Name, ast.Attribute)):
                    raise ValueError('indirect call forbidden')
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(isinstance(target, (ast.Attribute, ast.Subscript)) for target in targets):
                    raise ValueError('Python object mutation forbidden; use tl.store for output')
    return tree


def load_source(path):
    path = Path(path)
    check_source(path.read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('candidate_kernels', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
