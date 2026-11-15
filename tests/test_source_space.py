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


@pytest.mark.parametrize('field,value', [('grid',[0]), ('grid',['__import__("os")']), ('args',['__class__']), ('num_warps',True)])
def test_launch_interface_does_not_evaluate_expressions(field,value):
    _,config=baseline(cases()[0])
    config['launches'][0][field]=value
    with pytest.raises(ValueError): validate_plan(config)


def test_wrong_arity_fails_before_gpu():
    source,config=baseline(cases()[0])
    config['launches'][0]['args'].pop()
    with pytest.raises(ValueError,match='argument count'): _body_shapes(source,config)


def test_cohort_allows_family_baselines_but_rejects_within_family_drift():
    from kernel_study.analysis import validate_cohort
    a={'manifest':{'protocol_hash':'p','knowledge_hash':'k','provider':'llm','model':'m',
                  'environment':{'gpu':'gpu'},'case':{'family':'softmax','id':'s'},
                  'case_hash':'s','baseline_hash':'softmax-source'},'resolved_model':'m'}
    b=copy.deepcopy(a)
    b['manifest'].update(case={'family':'matmul','id':'m'},case_hash='m',baseline_hash='matmul-source')
    validate_cohort([a,b])
    drift=copy.deepcopy(a);drift['manifest']['baseline_hash']='changed'
    with pytest.raises(ValueError,match='Incompatible'): validate_cohort([a,b,drift])
    drift=copy.deepcopy(a);drift['manifest']['case_hash']='changed'
    with pytest.raises(ValueError,match='Incompatible'): validate_cohort([a,b,drift])
    drift=copy.deepcopy(a);drift['manifest']['code_hashes']={'kernel_study/search.py':'changed'}
    with pytest.raises(ValueError,match='Incompatible'): validate_cohort([a,b,drift])


def test_source_search_rejects_null_source_and_counts_failed_attempts(tmp_path,monkeypatch):
    import json
    import kernel_study.search as module
    from kernel_study.core import ROOT, read_json
    from kernel_study.providers import Reply
    contexts=[]
    class NullSourceProvider:
        name='test'; model='test'
        def complete(self,system,context,stage,seed):
            assert 'matmul_tensor' not in context['incumbent']['source']
            contexts.append((stage,copy.deepcopy(context)))
            answer=({'config':context['incumbent']['config'],'source':None} if stage=='implement' else {'note':'test'})
            return Reply(json.dumps(answer),1,1,'test',{})
    monkeypatch.setattr(module,'environment',lambda:{'gpu':'test'})
    workers=[]
    def worker(request,*args):
        workers.append(request)
        return {'status':'ok','confirmed':False,'paired_speedup':1.}
    monkeypatch.setattr(module,'run_worker',worker)
    protocol=read_json(ROOT/'configs/protocol.json')
    protocol.update(search_space='source-v1',candidate_cap=2,call_cap=8)
    result=module.search(cases()[0],'multi',11,protocol,NullSourceProvider(),tmp_path/'run')
    assert result['budget']['attempts']==2 and result['budget']['calls']==8
    assert len(workers)==2  # baseline and final only: no invalid proposal reaches GPU
    assert all('source_edit_required' in x['result']['error'] for x in result['history'])
    assert result['speedup']==1 and result['final']['candidate_id']==0
    assert contexts[-1][1]['proposed_candidate']['source'] is None
    assert 'previous_proposal' in contexts[4][1]
