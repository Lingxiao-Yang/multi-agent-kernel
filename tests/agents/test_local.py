from kernel_study.core import ROOT, read_json
from kernel_study.search import search
from kernel_study.providers import ScriptedProvider
import copy
from kernel_study.core import cases


class QuotaRejectedProvider:
    name='llm'
    model='test-model'
    spent=0
    completed_cost=0
    def complete(self,*args,**kwargs):
        raise RuntimeError('HTTP 429: credit_balance_exhausted')


def test_api_failure_is_not_completed_scientific_search(tmp_path,monkeypatch):
    import kernel_study.search as module
    monkeypatch.setattr(module,'environment',lambda:{'gpu':'test'})
    monkeypatch.setattr(module,'run_worker',lambda *a,**k:{'status':'ok','confirmed':False,'paired_speedup':1,
        'candidate':{'median_ms':1},'incumbent':{'median_ms':1},'library':{'median_ms':1}})
    case={'id':'test','family':'softmax','shape':[8,32],'dtype':'float32','atol':1e-5,'rtol':1e-5}
    r=search(case,'single',11,read_json(ROOT/'configs/protocol.json'),QuotaRejectedProvider(),tmp_path/'run')
    assert r['status']=='interrupted'
    assert not r['scientific_evidence']
    assert r['budget']['calls']==1
    assert r['budget']['tokens']==0
    assert r['estimated_inference_usd']==0


def fake_result(config, request):
    # Deliberately wrong candidates must never be retained.
    if config['fused']:
        return {'status':'incorrect','error':'injected wrong output'}
    return {'status':'ok','confirmed':False,'paired_speedup':1.,
            'candidate':{'median_ms':1.},'incumbent':{'median_ms':1.},'library':{'median_ms':1.}}


def test_incorrect_candidates_keep_baseline_and_count_calls(tmp_path,monkeypatch):
    import kernel_study.search as module
    monkeypatch.setattr(module,'environment',lambda:{'gpu':'test-only'})
    monkeypatch.setattr(module,'run_worker',lambda request,*args:fake_result(request['config'],request))
    protocol = read_json(ROOT/'configs/protocol.json')
    protocol['candidate_cap']=2
    case={'id':'test','family':'softmax','shape':[8,32],'dtype':'float32','atol':1e-5,'rtol':1e-5}
    result=search(case,'multi_knowledge',11,protocol,ScriptedProvider(),tmp_path/'run')
    assert result['budget']['attempts']==2
    assert result['budget']['calls']==8
    assert result['final']['candidate_id']==0
    assert result['speedup']==1
    assert not result['scientific_evidence']
    assert all(not row['accepted'] for row in result['history'])


def test_fresh_final_validation_falls_back(tmp_path,monkeypatch):
    import kernel_study.search as module
    monkeypatch.setattr(module,'environment',lambda:{'gpu':'test-only'})
    def worker(request,*args):
        if request.get('final') and request['config']['fused']:
            return {'status':'incorrect','error':'held-out failure'}
        return {'status':'ok','confirmed':request['config']['fused'],'paired_speedup':2.,
                'candidate':{'median_ms':1.},'incumbent':{'median_ms':2.},'library':{'median_ms':1.}}
    monkeypatch.setattr(module,'run_worker',worker)
    protocol=read_json(ROOT/'configs/protocol.json'); protocol['candidate_cap']=1
    case={'id':'test','family':'softmax','shape':[8,32],'dtype':'float32','atol':1e-5,'rtol':1e-5}
    result=search(case,'single',11,protocol,ScriptedProvider(),tmp_path/'run')
    assert result['history'][0]['accepted']
    assert result['final']['candidate_id']==0
    assert result['speedup']==1


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
