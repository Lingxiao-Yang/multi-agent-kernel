import sys
from types import SimpleNamespace

import pytest

from kernel_study import bench, metrics
from kernel_study.core import ROOT, read_json


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


def test_archived_cross_round_comparison_is_identified():
    old = {'candidate': {'median_ms': 2.}, 'library': {'median_ms': 12.}}
    report = metrics.library_timings(old, 1)
    assert report['library_over_candidate'] == 6.
    assert report['library_candidate_ms'] is None
    assert report['library_timing_basis'] == 'historical-unpaired'
    with pytest.raises(KeyError):
        metrics.library_timings(old, 2)


@pytest.mark.parametrize('mode', ['config', 'source-v1'])
def test_timed_forward_returns_fresh_output(mode, tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from kernel_study import source_space
    from kernel_study.core import baseline_config, cases
    case = next(c for c in cases() if c['id'] == 'softmax_irregular')
    inputs = bench.make_inputs(case, 71)
    config, path = baseline_config('softmax'), None
    if mode == 'source-v1':
        source, config = source_space.baseline(case)
        path = tmp_path / 'baseline.py'
        path.write_text(source)
    forward, _ = bench.prepare(case, config, inputs, path, fresh_output=True)
    first, second = forward(), forward()
    assert first.data_ptr() != second.data_ptr()
    for output in (first, second):
        torch.testing.assert_close(output, bench.reference(case, inputs), atol=case['atol'], rtol=case['rtol'])
