import ast
from graphlib import TopologicalSorter
from importlib.util import resolve_name
from pathlib import Path

import pytest


@pytest.fixture(scope='module')
def imports():
    root = Path(__file__).resolve().parents[1]
    modules = {}
    for path in (root/'kernel_study').rglob('*.py'):
        parts = path.relative_to(root).with_suffix('').parts
        package = path.name=='__init__.py'
        name = '.'.join(parts[:-1] if package else parts)
        modules[name] = (ast.parse(path.read_text()), name if package else name.rpartition('.')[0])
    edges = []
    for name, (tree, package) in modules.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = resolve_name('.'*node.level + (node.module or ''), package)
                targets = [f'{base}.{alias.name}' if f'{base}.{alias.name}' in modules else base for alias in node.names]
            else:
                continue
            for target in targets:
                if target in modules:
                    edges.append((name, target, node in tree.body))
    return edges


def test_package_imports_are_acyclic(imports):
    graph = TopologicalSorter()
    for source, target, _ in imports:
        graph.add(source, target)
    tuple(graph.static_order())


def test_internal_imports_are_at_module_scope(imports):
    assert not [(source, target) for source, target, at_top in imports if not at_top]


def test_imports_follow_the_study_layers(imports):
    layers = {'records':0, 'budget':0, 'providers':0, 'tasks':1, 'judge':2,
              'agents':3, 'campaign':4, 'cli':5}
    def layer(name):
        parts = name.split('.')
        return 0 if len(parts)==1 else layers[parts[1]]

    for source, target, _ in imports:
        assert layer(target) <= layer(source), f'{source} imports {target}'
