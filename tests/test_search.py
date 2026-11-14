from pathlib import Path
from kernel_study.core import ROOT, read_json
from kernel_study.providers import ScriptedProvider
from kernel_study.search import search


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
