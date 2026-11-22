from kernel_study import bench


def test_public_requests_reach_same_referee(monkeypatch):
    from kernel_study import public_cases
    seen = []
    monkeypatch.setattr(public_cases, 'evaluate', lambda r: seen.append(r) or {'status':'ok'})
    for interface in ('native','declarative'):
        request = {'case':{'public_id':4},'interface':interface}
        assert bench.evaluate(request)['status'] == 'ok'
    assert [r['interface'] for r in seen] == ['native','declarative']
