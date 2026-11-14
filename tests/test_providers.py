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
