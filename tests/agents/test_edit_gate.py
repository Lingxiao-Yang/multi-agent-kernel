import copy

import pytest

from kernel_study.agents.edit_gate import check_edit
from kernel_study.tasks.kernelbench import experiment
from kernel_study.tasks.starts import start


def test_parameter_only_cosmetic_and_constant_only_edits_rejected():
    source, config = start(experiment()['cases']['2'])
    changed = copy.deepcopy(config)
    changed['launches'][0]['num_warps'] = 8
    for proposal in [source, source.replace('acc', 'accumulator'), source.replace(', 0, tl.float32', ', 1, tl.float32')]:
        with pytest.raises(ValueError, match='source_edit_required'):
            check_edit(proposal, changed, source, config)


def test_straight_line_temporary_rewrites_are_not_source_optimization(public_case):
    source, config = start(public_case)
    changed = source.replace('x = tl.load(X+i,', 'pointer = X+i\n    x = tl.load(pointer,')
    changed = changed.replace('tl.store(Y+i,0.5*x*(1.+tl.erf(x*0.7071067811865476)),',
                              'value = 0.5*x*(1.+tl.erf(x*0.7071067811865476))\n    tl.store(Y+i,value,')
    assert changed != source
    with pytest.raises(ValueError, match='source_edit_required'):
        check_edit(changed, config, source, config)


def test_device_schedule_change_allowed_and_missing_entry_rejected():
    source, config = start(experiment()['cases']['2'])
    changed_source = source.replace('range(triton.cdiv(K, BK))', 'tl.range(0, triton.cdiv(K, BK), loop_unroll_factor=2)')
    assert check_edit(changed_source,config,source,config)['structural_edit']
    changed = copy.deepcopy(config)
    changed['launches'][0]['kernel'] = 'missing'
    with pytest.raises(ValueError, match='missing kernel'):
        check_edit(changed_source,changed,source,config)
