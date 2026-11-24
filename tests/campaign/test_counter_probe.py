import runpy
import sys
from types import SimpleNamespace

from kernel_study.records import ROOT


def test_counter_probe_profiles_one_warmed_public_launch(monkeypatch):
    calls = []
    cuda = SimpleNamespace(synchronize=lambda: None, cudart=lambda: SimpleNamespace(
        cudaProfilerStart=lambda: calls.append('start'), cudaProfilerStop=lambda: calls.append('stop')))
    torch = SimpleNamespace(cuda=cuda,float32='fp32',zeros=lambda *a,**kw: object(),empty_like=lambda x: object())
    class Kernel:
        def __getitem__(self, grid):
            assert grid==(1,)
            return lambda x, y, total, block, **kw: calls.append((total,block))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'kernel_study.tasks.baselines.gelu',SimpleNamespace(gelu=Kernel()))
    runpy.run_path(str(ROOT/'scripts/profile_target.py'),run_name='__main__')
    assert calls==[(2048,2048),'start',(2048,2048),'stop']
