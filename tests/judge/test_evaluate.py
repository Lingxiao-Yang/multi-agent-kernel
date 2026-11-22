import sys

from kernel_study import worker
from kernel_study.core import read_json, write_json


def test_public_requests_reach_same_referee(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(worker, 'evaluate', lambda r: seen.append(r) or {'status':'ok'})
    request, output = tmp_path/'request.json', tmp_path/'output.json'
    monkeypatch.setattr(sys, 'argv', ['worker', str(request), str(output)])
    for interface in ('native','declarative'):
        write_json(request, {'case':{'public_id':4},'interface':interface})
        worker.main()
        assert read_json(output)['status'] == 'ok'
    assert [r['interface'] for r in seen] == ['native','declarative']
