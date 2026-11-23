"""Check the native module before either search can execute a proposed wrapper."""
import ast


class StaticRejection(ValueError):
    pass


def reject(node, reason):
    raise StaticRejection(f'line {getattr(node, "lineno", 1)}: {reason}')


def dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted(node.value) + '.' + node.attr
    return ''


def tensor(kind):
    return kind == 'input' or isinstance(kind, ast.AST)


def target_names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def plain_signature(fn, modules):
    a = fn.args
    if a.vararg or a.kwarg or any(x.arg in modules for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)):
        reject(fn, 'use named kernel arguments')
    for value in [*a.defaults, *a.kw_defaults]:
        if value is not None and not isinstance(value, ast.Constant):
            reject(value, 'defaults must be immutable literals')
    for annotation in [fn.returns, *(x.annotation for x in (*a.posonlyargs, *a.args, *a.kwonlyargs))]:
        if annotation is not None and dotted(annotation) not in ('torch.Tensor','tl.constexpr','int','float','bool'):
            reject(annotation, 'unsupported signature annotation')


def device_writes(functions):
    """Find the formal pointers used as store destinations, including through JIT helpers."""
    writes = {name:set() for name in functions}
    for fn in functions.values():
        locals_ = {n.id for n in ast.walk(fn) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        locals_.update(a.arg for a in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs))
        for node in ast.walk(fn):
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                                 ast.With, ast.Try, ast.Delete, ast.Lambda, ast.Yield)):
                reject(node, 'host or persistent state in a device function')
            if isinstance(node, ast.Name) and (node.id.startswith('__') or node.id=='torch'):
                reject(node, 'host object in a device function')
            if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
                reject(node, 'private device attribute')
            if isinstance(node, ast.Call):
                func = dotted(node.func)
                if isinstance(node.func, ast.Name):
                    if func not in functions and func not in ('range','int','float','bool','min','max','abs'):
                        reject(node, 'unsupported device call '+func)
                elif not isinstance(node.func, ast.Attribute):
                    reject(node, 'indirect device call')
                elif isinstance(node.func,ast.Attribute):
                    root = func.split('.')[0]
                    if root and root not in locals_ and not (func.startswith(('tl.','triton.language.'))
                                                              or func in ('triton.cdiv','triton.next_power_of_2','triton.jit')):
                        reject(node, 'unsupported device call '+func)
    changed = True
    while changed:
        changed = False
        for name, fn in functions.items():
            params = [a.arg for a in (*fn.args.posonlyargs,*fn.args.args,*fn.args.kwonlyargs)]
            aliases = {p:{p} for p in params}
            def pointers(expr):
                if isinstance(expr, ast.Name):
                    return aliases.get(expr.id, set())
                if isinstance(expr, ast.Call) and dotted(expr.func) in ('tl.load','triton.language.load'):
                    return set()
                return set().union(*(pointers(child) for child in ast.iter_child_nodes(expr)))
            assignments = [n for n in ast.walk(fn) if isinstance(n, (ast.Assign,ast.AnnAssign,ast.AugAssign)) and n.value is not None]
            # Pointer aliases can refer to names different branches assigned earlier.
            while True:
                grew = False
                for assignment in assignments:
                    targets = assignment.targets if isinstance(assignment,ast.Assign) else [assignment.target]
                    for target in targets:
                        if not isinstance(target,(ast.Name,ast.Tuple)):
                            reject(target, 'device assignments must target local names')
                        for var in target_names(target):
                            old = aliases.setdefault(var,set())
                            extra = pointers(assignment.value) - old
                            if extra:
                                old.update(extra)
                                grew = True
                if not grew:
                    break
            for call in (n for n in ast.walk(fn) if isinstance(n,ast.Call)):
                func = dotted(call.func)
                destinations = []
                if func in functions:
                    other = functions[func].args
                    bound = dict(zip([a.arg for a in (*other.posonlyargs,*other.args)],call.args))
                    bound.update((kw.arg,kw.value) for kw in call.keywords)
                    destinations.extend(bound[p] for p in writes[func] if p in bound)
                elif (func in ('tl.store','triton.language.store','tl.descriptor_store','triton.language.descriptor_store')
                      or func.startswith(('tl.atomic_','triton.language.atomic_'))):
                    destinations.extend(call.args[:1] or [kw.value for kw in call.keywords if kw.arg=='pointer'])
                elif isinstance(call.func,ast.Attribute) and call.func.attr=='store':
                    destinations.append(call.func.value)
                for dest in destinations:
                    extra = pointers(dest) - writes[name]
                    if extra:
                        writes[name].update(extra)
                        changed = True
    return writes


class Wrapper:
    def __init__(self, functions, writes, constants):
        self.functions = functions
        self.writes = writes
        self.constants = constants

    def expression(self, node, scope):
        if isinstance(node,ast.Name):
            if node.id in scope:
                return scope[node.id]
            if node.id in self.constants:
                return 'scalar'
            reject(node,'unknown wrapper name '+node.id)
        if isinstance(node,ast.Constant):
            return 'scalar'
        if isinstance(node,ast.Attribute):
            if dotted(node) in {'torch.'+dtype for dtype in (
                    'bfloat16','float16','float32','float64','bool','uint8','int8','int16','int32','int64')}:
                return 'scalar'
            kind = self.expression(node.value,scope)
            if tensor(kind) and node.attr in ('shape','ndim','dtype','device','is_cuda'):
                return 'scalar'
            reject(node,'unsupported wrapper attribute '+node.attr)
        if isinstance(node,ast.Call):
            return self.call(node,scope)
        if isinstance(node,ast.Lambda):
            grid = dict(scope, **{arg.arg:'scalar' for arg in node.args.args})
            if (len(node.args.args) != 1 or node.args.posonlyargs or node.args.defaults or node.args.kwonlyargs
                    or node.args.vararg or node.args.kwarg or self.expression(node.body,grid)!='scalar'):
                reject(node,'invalid launch grid')
            return 'grid'
        if isinstance(node,(ast.BinOp,ast.UnaryOp,ast.BoolOp,ast.Compare,ast.Tuple,ast.List,ast.Dict,
                            ast.Subscript,ast.Slice,ast.JoinedStr,ast.FormattedValue,ast.IfExp)):
            values = [self.expression(child,scope) for child in ast.iter_child_nodes(node)
                      if isinstance(child,(ast.expr,ast.Slice))]
            if all(v=='scalar' for v in values):
                return 'scalar'
            reject(node,'tensor computation is only allowed in Triton kernels')
        reject(node,'unsupported wrapper expression '+type(node).__name__)

    def call(self,node,scope):
        f = node.func
        args = [self.expression(a,scope) for a in node.args]
        kwargs = {kw.arg:self.expression(kw.value,scope) for kw in node.keywords}
        if None in kwargs or 'out' in kwargs:
            reject(node,'unsupported output or expanded keyword arguments')
        if isinstance(f,ast.Subscript) and isinstance(f.value,ast.Name) and f.value.id in self.functions:
            name = f.value.id
            if self.expression(f.slice,scope) not in ('scalar','grid'):
                reject(f,'invalid launch grid')
            params = self.functions[name].args
            bindings = dict(zip([a.arg for a in (*params.posonlyargs,*params.args)],args))
            bindings.update(kwargs)
            for param in self.writes[name]:
                if bindings.get(param)=='input':
                    reject(node,'kernel writes to input '+param)
            return frozenset(bindings[p] for p in self.writes[name] if isinstance(bindings.get(p), ast.AST))
        func = dotted(f)
        if func in ('torch.empty','torch.zeros','torch.ones','torch.full','torch.empty_like','torch.zeros_like',
                    'torch.ones_like','torch.empty_strided'):
            if func.endswith('_like'):
                if not args or not tensor(args[0]) or any(a!='scalar' for a in args[1:]):
                    reject(node,'invalid allocation arguments')
            elif any(a!='scalar' for a in args):
                reject(node,'allocation shape must be metadata')
            if any(a!='scalar' for a in kwargs.values()):
                reject(node,'allocation options must be metadata')
            return node
        if isinstance(f,ast.Attribute) and f.attr in ('stride','size','numel','dim','is_contiguous',
                                                     'view','unsqueeze','squeeze','new_empty','new_zeros','new_full'):
            kind = self.expression(f.value,scope)
            if not tensor(kind) or any(a!='scalar' for a in [*args,*kwargs.values()]):
                reject(node,'invalid tensor metadata or allocation call')
            if f.attr.startswith('new_'):
                return node
            return kind if f.attr in ('view','squeeze','unsqueeze') else 'scalar'
        if func in ('triton.cdiv','triton.next_power_of_2','tuple','len','int','float','min','max','range',
                    'ValueError','RuntimeError','AssertionError') and all(a=='scalar' for a in [*args,*kwargs.values()]):
            return 'scalar'
        reject(node,'unsupported wrapper call '+(func or type(f).__name__))

    def block(self, body, scope, written, grid=False):
        def assign(target, kind):
            if isinstance(target,ast.Name) and target.id not in {'torch','tl','triton',*self.functions}:
                scope[target.id] = kind
            elif isinstance(target,(ast.Tuple,ast.List)) and kind=='scalar':
                for item in target.elts:
                    assign(item,kind)
            else:
                reject(target,'assign only to local names')
        for node in body:
            if isinstance(node,(ast.Assign,ast.AnnAssign)):
                if node.value is None:
                    reject(node,'local assignment requires a value')
                kind = self.expression(node.value,scope)
                if isinstance(kind,frozenset):
                    reject(node,'use a standalone kernel launch')
                targets = node.targets if isinstance(node,ast.Assign) else [node.target]
                for target in targets:
                    assign(target,kind)
            elif isinstance(node,ast.If):
                if self.expression(node.test,scope)!='scalar':
                    reject(node.test,'condition must use metadata')
                branches = []
                for statements in (node.body,node.orelse):
                    local, stored = scope.copy(), written.copy()
                    if self.block(statements, local, stored, grid):
                        branches.append((local,stored))
                if not branches:
                    return False
                scope.clear()
                written.clear()
                written.update(set.intersection(*(stored for _,stored in branches)))
                for key in set.intersection(*(set(local) for local,_ in branches)):
                    values = {local[key] for local,_ in branches}
                    if len(values)==1:
                        scope[key] = values.pop()
                    elif all(tensor(v) for v in values):
                        scope[key] = 'input' if 'input' in values else ast.Name(id=key)
                    else:
                        scope[key] = 'ambiguous'
                    if all(local[key] in stored for local,stored in branches):
                        written.add(scope[key])
            elif isinstance(node,ast.FunctionDef) and not grid and not node.decorator_list:
                plain_signature(node,{'torch','triton','tl'})
                nested = dict(scope, **{arg.arg:'scalar' for arg in node.args.args})
                if self.block(node.body,nested,set(),grid=True):
                    reject(node,'grid must return launch dimensions')
                assign(ast.Name(id=node.name), 'grid')
            elif isinstance(node,ast.Return):
                value = self.expression(node.value,scope)
                if grid and value!='scalar':
                    reject(node,'invalid launch grid')
                if not grid and value not in written:
                    reject(node,'return a fresh output written by a kernel launch')
                return False
            elif isinstance(node,ast.Assert):
                if self.expression(node.test,scope)!='scalar':
                    reject(node,'assertion must use metadata')
                if node.msg is not None:
                    self.expression(node.msg,scope)
            elif isinstance(node,ast.Raise) and node.exc is not None:
                if node.cause or self.expression(node.exc,scope)!='scalar':
                    reject(node,'invalid argument check')
                return False
            elif isinstance(node,ast.Expr):
                kind = self.expression(node.value, scope)
                if isinstance(kind, frozenset) and not grid:
                    written.update(kind)
                elif not (isinstance(node.value,ast.Constant) and isinstance(node.value.value,str)):
                    reject(node,'expected a kernel launch')
            else:
                reject(node,'unsupported wrapper statement '+type(node).__name__)
        return True


def check_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise StaticRejection(f'line {exc.lineno}: {exc.msg}') from exc
    functions, constants = {}, set()
    host = None
    modules = {'torch','triton','tl'}
    for node in tree.body:
        if isinstance(node,ast.Import):
            for a in node.names:
                if (a.name,a.asname) not in (('torch',None),('triton',None),('triton.language','tl')):
                    reject(node,'import torch, triton, or triton.language as tl')
        elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            continue
        elif isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant):
            for target in node.targets:
                if not isinstance(target,ast.Name) or target.id in modules | constants | functions.keys() | {'kernel_function'}:
                    reject(node,'invalid module constant')
                constants.add(target.id)
        elif isinstance(node,ast.FunctionDef) and node.name not in modules | constants | functions.keys():
            plain_signature(node,modules)
            if node.name=='kernel_function' and not node.decorator_list and host is None:
                host=node
            elif node.name!='kernel_function' and [dotted(d) for d in node.decorator_list]==['triton.jit']:
                functions[node.name]=node
            else:
                reject(node,'expected kernel_function or a Triton JIT kernel')
        else:
            reject(node,'module may contain only imports, immutable constants and functions')
    if host is None or not functions:
        raise StaticRejection('expected kernel_function and a Triton JIT kernel')
    writes = device_writes(functions)
    scope = {p.arg:'input' for p in (*host.args.posonlyargs,*host.args.args,*host.args.kwonlyargs)}
    if Wrapper(functions,writes,constants).block(host.body,scope,set()):
        reject(host,'kernel_function must return an output written by a kernel launch')
    return tree
