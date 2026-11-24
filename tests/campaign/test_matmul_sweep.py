import ast
from itertools import product

from kernel_study.tasks.starts import start


def test_matmul_grid_covers_the_same_output_with_the_frozen_ieee_kernel():
    from scripts import sweep_matmul as sweep
    source = start({'family':'matmul'})
    grid = sweep.configurations()
    assert grid[0]==sweep.BASELINE and sweep.variant(source,sweep.BASELINE)==source
    assert len(set(grid))==len(grid)==72
    assert set(grid)==set(product((32,64,128),(32,64,128),(16,32),(4,8),(1,3)))
    for bm,bn,bk,warps,stages in grid:
        tree = ast.parse(sweep.variant(source,(bm,bn,bk,warps,stages)))
        assert ast.dump(tree.body[2])==ast.dump(ast.parse(source).body[2])
        launch = tree.body[-1].body[-2].value
        assert [v.value for v in launch.func.slice.elts]==[2048//bm,4096//bn]
        assert [v.value for v in launch.args[-3:]]==[bm,bn,bk]
        assert {v.arg:v.value.value for v in launch.keywords}=={'num_warps':warps,'num_stages':stages}
