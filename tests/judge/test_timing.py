import sys
from types import SimpleNamespace
from kernel_study import bench, metrics
from kernel_study.core import ROOT, read_json


def test_noise_rule_requires_two_wins_and_uses_calibration_only():
    from kernel_study.replay import threshold_from_controls, confirmed
    calibration = [[1.2, 1.04], [0.99, 1.13], [1.02, 1.03]]
    threshold = threshold_from_controls(calibration, 1.01)
    assert threshold == 1.04
    assert not confirmed([1.08, 1.04], threshold)
    assert confirmed([1.06, 1.05], threshold)


def test_library_comparison_uses_its_own_candidate_samples(monkeypatch):
    cuda = SimpleNamespace(set_device=lambda index: None, synchronize=lambda: None)
    torch = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(bench, 'environment', lambda: {})
    monkeypatch.setattr(bench, 'validate', lambda *args: ([], 0.))
    monkeypatch.setattr(bench, 'make_inputs', lambda *args: [])
    monkeypatch.setattr(bench, 'prepare', lambda *args, **kwargs: (lambda: None, []))
    timings = iter([(2., 2.), (4., 12.)])
    monkeypatch.setattr(bench, 'time_pair', lambda *args: [
        {'median_ms': value, 'samples_ms': [value]} for value in next(timings)])
    protocol = read_json(ROOT / 'configs/protocol.json')
    result = bench.evaluate({'case': {'family': 'softmax', 'dtype': 'float32', 'shape': [2, 4]},
                             'protocol': protocol, 'config': {}, 'library': True})
    assert result['candidate']['samples_ms'] == [2.]
    assert result['library_candidate']['samples_ms'] == [4.]
    report = metrics.library_timings(result, protocol['schema_version'])
    assert report['library_candidate_ms'] == 4.
    assert report['library_over_candidate'] == 3.
    assert report['library_timing_basis'] == 'paired'
