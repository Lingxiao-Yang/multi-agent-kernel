import copy
import pytest
from kernel_study.core import cases, validate_config
from kernel_study.source_space import baseline, check_edit, validate_plan, _body_shapes


def test_only_baseline_functions_are_exposed_and_all_launches_match():
    for case in cases():
        source, config = baseline(case)
        validate_config(config)
        _body_shapes(source, config)
        for forbidden in ['matmul_tensor', 'softmax_fused', 'layernorm_fused', 'bias_relu']:
            assert forbidden not in source


def test_parameter_only_cosmetic_and_constant_only_edits_rejected():
    case = next(c for c in cases() if c['family']=='matmul')
    source, config = baseline(case)
    changed = copy.deepcopy(config)
    changed['launches'][0]['num_warps'] = 8
    for proposal in [source, source.replace('acc', 'accumulator'), source.replace('0, tl.float32', '1, tl.float32')]:
        with pytest.raises(ValueError, match='source_edit_required'):
            check_edit(proposal, changed, source, config)


def test_device_algorithm_change_allowed_and_missing_entry_rejected():
    case = next(c for c in cases() if c['family']=='matmul')
    source, config = baseline(case)
    changed_source = source.replace('acc +=', 'acc = acc +')
    assert check_edit(changed_source,config,source,config)['structural_edit']
    # The screen is intentionally not a proof of semantic difference.
    changed = copy.deepcopy(config)
    changed['launches'][0]['kernel'] = 'missing'
    with pytest.raises(ValueError, match='missing kernel'):
        check_edit(changed_source,changed,source,config)


@pytest.mark.parametrize('field,value', [('grid',[0]), ('grid',['__import__("os")']), ('args',['__class__']), ('num_warps',True)])
def test_launch_interface_does_not_evaluate_expressions(field,value):
    _,config=baseline(cases()[0])
    config['launches'][0][field]=value
    with pytest.raises(ValueError): validate_plan(config)


def test_wrong_arity_fails_before_gpu():
    source,config=baseline(cases()[0])
    config['launches'][0]['args'].pop()
    with pytest.raises(ValueError,match='argument count'): _body_shapes(source,config)
