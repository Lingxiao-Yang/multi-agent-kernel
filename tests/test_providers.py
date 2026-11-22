import io
import json
import urllib.error
import pytest
from kernel_study.providers import ChatProvider


def test_rejected_quota_request_releases_financial_reservation(monkeypatch):
    def reject(*args,**kwargs):
        body=json.dumps({'error':{'code':'credit_balance_exhausted','message':'No credits.'}}).encode()
        raise urllib.error.HTTPError('https://api.openai.com/v1/chat/completions',429,'Too Many Requests',{},io.BytesIO(body))
    monkeypatch.setattr('urllib.request.urlopen',reject)
    p=ChatProvider('fixed-model','https://api.openai.com/v1',16,input_price=.4,output_price=1.6,dollar_cap=1)
    with pytest.raises(RuntimeError,match='credit_balance_exhausted'):
        p.complete('Return JSON',{},'analyze',11)
    assert p.spent==0
    assert p.completed_cost==0


def test_dollar_cap_prevents_request(monkeypatch):
    def must_not_send(*args,**kwargs):
        raise AssertionError('request was sent despite cap')
    monkeypatch.setattr('urllib.request.urlopen',must_not_send)
    p=ChatProvider('fixed-model','https://api.openai.com/v1',4096,input_price=1,output_price=1,dollar_cap=0)
    with pytest.raises(RuntimeError,match='dollar_cap_reservation'):
        p.complete('Return JSON',{},'analyze',11)


def test_reported_usage_replaces_conservative_cost_reservation(monkeypatch):
    data={'model':'fixed-model','id':'test','usage':{'prompt_tokens':100,'completion_tokens':50},
          'choices':[{'message':{'content':'{"ok":true}'}}]}
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**k:io.BytesIO(json.dumps(data).encode()))
    p=ChatProvider('fixed-model','https://api.openai.com/v1',4096,input_price=1,output_price=2,dollar_cap=1)
    r=p.complete('Return JSON',{},'analyze',11)
    assert r.input_tokens==100 and r.output_tokens==50
    assert p.spent==pytest.approx(.0002)
    assert p.completed_cost==pytest.approx(.0002)


def test_transport_failure_is_logged_and_upstream_cannot_silently_call_again(tmp_path,monkeypatch):
    import urllib.request
    import urllib.error
    from kernel_study.reference_search import Ledger,MeteredChat
    def offline(*args,**kwargs): raise urllib.error.URLError('unreachable')
    monkeypatch.setattr(urllib.request,'urlopen',offline)
    ledger = Ledger(tmp_path, 'sample', 4, 16, 1., 1., 1.)
    chat = MeteredChat(ledger,tmp_path,model='pinned',endpoint='http://localhost:1')
    with pytest.raises(Exception): chat.send_messages([], 'proposal', 1)
    with pytest.raises(Exception): chat.send_messages([], 'repair', 2)
    assert ledger.summary()['calls']==1 and ledger.summary()['committed_usd']>0
    failures = [e for e in map(json.loads,(tmp_path/'model_events.jsonl').read_text().splitlines()) if e['event']=='request_error']
    assert [e['reason'] for e in failures]==['transport']


def test_both_model_interfaces_meter_the_wire_payload(tmp_path, monkeypatch):
    import io
    import json
    import urllib.request
    from kernel_study.reference_search import Ledger, MeteredChat
    sent = []
    def response(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'model':'pinned','choices':[{'message':{'content':'{}'}}],
            'usage':{'prompt_tokens':20,'completion_tokens':7}}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',response)
    ledger = Ledger(tmp_path,'pair',2,2,1.,1.,1.)
    provider = MeteredChat(ledger,tmp_path,model='pinned',endpoint='http://localhost:1',
                          max_output_tokens=100,input_price=1.,output_price=1.)
    provider.complete('single',{},'analyze',11)
    messages = [{'role':'user','content':'native bottleneck diagnosis'}]
    provider.send_messages(messages,'proposal',12,max_tokens=25)
    assert sent[0]['response_format']=={'type':'json_object'}
    assert 'response_format' not in sent[1] and sent[1]['messages']==messages
    assert sent[1]['max_completion_tokens']==25
    with pytest.raises(RuntimeError): provider.send_messages(messages,'repair',13)
    assert len(sent)==2 and ledger.summary()['input_tokens']==40
