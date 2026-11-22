import sys
from types import SimpleNamespace
from contextlib import nullcontext
from kernel_study import bench, core, public_cases
from kernel_study.core import ROOT, read_json


def test_noise_rule_requires_two_wins_and_uses_calibration_only():
    from kernel_study.replay import threshold_from_controls, confirmed
    calibration = [[1.2, 1.04], [0.99, 1.13], [1.02, 1.03]]
    threshold = threshold_from_controls(calibration, 1.01)
    assert threshold == 1.04
    assert not confirmed([1.08, 1.04], threshold)
    assert confirmed([1.06, 1.05], threshold)


def test_library_comparison_uses_its_own_candidate_samples(monkeypatch, public_case):
    torch = SimpleNamespace(no_grad=nullcontext, backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(core, 'environment', lambda: {})
    monkeypatch.setattr(public_cases, 'validate', lambda *args: ([], 0.))
    monkeypatch.setattr(public_cases, 'task_inputs', lambda *args: (lambda: None, [], []))
    monkeypatch.setattr(public_cases, 'bind', lambda *args: (lambda: None, []))
    timings = iter([(2., 2.), (4., 12.)])
    monkeypatch.setattr(bench, 'time_pair', lambda *args: [
        {'median_ms': value, 'samples_ms': [value]} for value in next(timings)])
    protocol = read_json(ROOT / 'configs/protocol.json')
    result = public_cases.evaluate(dict(case=public_case, protocol=protocol, config={}, source='candidate.py', library=True))
    assert result['candidate']['samples_ms'] == [2.]
    assert result['library_candidate']['samples_ms'] == [4.]
    assert result['library']['samples_ms'] == [12.]
