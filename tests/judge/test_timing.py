import sys
from contextlib import nullcontext
from types import SimpleNamespace

from kernel_study.judge import evaluate as worker, timing
from kernel_study.judge.timing import confirmed, threshold_from_controls
from kernel_study.records import ROOT, read_json


def test_noise_rule_requires_two_wins_and_uses_calibration_only():
    calibration = [[1.2, 1.04], [0.99, 1.13], [1.02, 1.03]]
    threshold = threshold_from_controls(calibration, 1.01)
    assert threshold == 1.04
    assert not confirmed([1.08, 1.04], threshold)
    assert confirmed([1.06, 1.05], threshold)


def test_library_comparison_uses_its_own_candidate_samples(monkeypatch, public_case):
    torch = SimpleNamespace(no_grad=nullcontext, backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace())))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setattr(worker, 'environment', lambda: {})
    monkeypatch.setattr(worker, 'validate', lambda *args: ([], 0.))
    monkeypatch.setattr(worker, 'task_inputs', lambda *args: (lambda: None, [], []))
    monkeypatch.setattr(worker, 'bind', lambda *args: lambda: None)
    timings = iter([(2., 2.), (4., 12.)])
    monkeypatch.setattr(timing, 'time_pair', lambda *args: [
        {'median_ms': value, 'samples_ms': [value]} for value in next(timings)])
    protocol = read_json(ROOT / 'configs/protocol.json')
    result = worker.evaluate(dict(case=public_case, protocol=protocol, config={}, source='candidate.py', library=True))
    assert result['candidate']['samples_ms'] == [2.]
    assert result['library_candidate']['samples_ms'] == [4.]
    assert result['library']['samples_ms'] == [12.]
