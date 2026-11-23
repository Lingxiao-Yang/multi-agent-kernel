from itertools import product

from kernel_study.records import read_json
from kernel_study.tasks.starts import start
from scripts import sweep_matvec as sweep


def test_grid_only_changes_the_four_launch_parameters():
    original = start({'family': 'matvec'})
    assert sweep.variant(original, sweep.BASELINE) == original
    order = sweep.configurations()
    assert order[0] == sweep.BASELINE and order == sweep.configurations()
    assert set(order) == set(product((1, 2, 4, 8, 16, 32, 64, 128), (64, 128, 256, 512),
                                     (1, 2, 4, 8), (1, 2, 3, 4)))
    assert len(order) == len(set(order))
    text = sweep.variant(original, (16, 128, 8, 4))
    assert tuple(line.strip() for line, prior in zip(text.splitlines(), original.splitlines()) if line != prior) == (
        'BLOCK_M = 16', 'BLOCK_K = 128', 'num_warps=8,', 'num_stages=4,')


def test_grid_keeps_failures_and_rechecks_finalists_on_new_inputs(tmp_path, monkeypatch):
    calls = []
    def worker(request, directory, label):
        calls.append((label, request))
        if 'm32-' in label:
            return {'status': 'compile_error', 'error': 'too many resources'}
        fast = 'm16-' in label
        return dict(status='ok', confirmed=fast, candidate={'median_ms':4 if fast else 10},
                    incumbent={'median_ms':10}, paired_speedup=2.5 if fast else 1.)
    monkeypatch.setattr(sweep, 'run_worker', worker)
    case = {'family':'matvec'}
    protocol = dict(validation_seeds=[101], final_validation_seeds=[10001,10002,10003,10004,10005])
    grid = [sweep.BASELINE, (16,256,8,4), (32,128,2,1)]
    result = sweep.run(tmp_path, case, protocol, grid)
    assert len([x for x in calls if not x[1].get('final')]) == 3
    finals = [q for _,q in calls if q.get('final')]
    assert len(finals) == 3
    assert [q['protocol']['final_validation_seeds'][0] for q in finals] == [10001,10002,10003]
    assert all(q['library'] for q in finals)
    assert result['best']['config'] == [16,256,8,4] and result['best']['speedup'] == 2.5
    assert result['counts'] == {'ok':2, 'compile_error':1}
    assert read_json(tmp_path/'summary.json') == result
