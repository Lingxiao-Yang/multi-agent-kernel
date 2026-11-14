from kernel_study.core import ROOT, read_json
from kernel_study.search import search


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
