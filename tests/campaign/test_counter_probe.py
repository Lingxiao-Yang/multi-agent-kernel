import runpy
import sys
from types import SimpleNamespace

from kernel_study.judge import interface
from kernel_study.records import ROOT
from kernel_study.tasks.starts import start


def test_counter_probe_profiles_one_warmed_public_launch(monkeypatch):
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: None, cudart=lambda: SimpleNamespace(
        cudaProfilerStart=lambda: calls.append('start'), cudaProfilerStop=lambda: calls.append('stop')))
    torch = SimpleNamespace(cuda=cuda, float32='float32', zeros=lambda *args, **kwargs: object())
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.delenv('KERNEL_BASELINE_UPSTREAM', raising=False)

    def forward(case, path, config, inputs):
        assert case['public_id'] == 26 and len(inputs) == 1
        assert (path.read_text(), config) == start(case)
        assert config['launches'][0]['grid'] == [1]
        return lambda: calls.append('launch')

    monkeypatch.setattr(interface, 'local_forward', forward)
    runpy.run_path(str(ROOT/'scripts/profile_target.py'), run_name='__main__')
    assert calls == ['launch', 'start', 'launch', 'stop']
