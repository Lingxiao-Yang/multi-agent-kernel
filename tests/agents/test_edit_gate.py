import copy
import pytest
from kernel_study.core import cases
from kernel_study.source_space import baseline, check_edit


def test_parameter_only_cosmetic_and_constant_only_edits_rejected():
    case = next(c for c in cases() if c['family']=='matmul')
    source, config = baseline(case)
    changed = copy.deepcopy(config)
    changed['launches'][0]['num_warps'] = 8
    for proposal in [source, source.replace('acc', 'accumulator'), source.replace('0, tl.float32', '1, tl.float32')]:
        with pytest.raises(ValueError, match='source_edit_required'):
            check_edit(proposal, changed, source, config)


def test_straight_line_temporary_rewrites_are_not_source_optimization():
    case=next(c for c in cases() if c['family']=='softmax')
    source,config=baseline(case)
    changed=source.replace('x = tl.load(X + row * N + col,',
                           'ptrs = row * N + col\n    x = tl.load(X + ptrs,')
    changed=changed.replace('denom = tl.sum(tl.exp(x - mx), 0)',
                            'exps = tl.exp(x - mx)\n    denom = tl.sum(exps, 0)')
    with pytest.raises(ValueError,match='source_edit_required'):
        check_edit(changed,config,source,config)


def test_device_schedule_change_allowed_and_missing_entry_rejected():
    case = next(c for c in cases() if c['family']=='matmul')
    source, config = baseline(case)
    changed_source = source.replace('range(K)', 'tl.range(0, K, loop_unroll_factor=2)')
    assert check_edit(changed_source,config,source,config)['structural_edit']
    # Explicit loop scheduling is device code; launch names must still resolve.
    changed = copy.deepcopy(config)
    changed['launches'][0]['kernel'] = 'missing'
    with pytest.raises(ValueError, match='missing kernel'):
        check_edit(changed_source,changed,source,config)
