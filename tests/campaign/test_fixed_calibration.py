import sys
from types import SimpleNamespace

from kernel_study.campaign import preflight


def test_public_calibration_counts_false_wins_at_the_fixed_threshold(tmp_path,monkeypatch):
    cuda = SimpleNamespace(reset_peak_memory_stats=lambda:None,max_memory_allocated=lambda:0,max_memory_reserved=lambda:0)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=cuda,
                         backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()))))
    task = tmp_path/'problem.py'
    task.write_text('problem')
    monkeypatch.setattr(preflight.kernelbench,'task_path',lambda case:task)
    monkeypatch.setattr(preflight.kernelbench,'task_inputs',lambda *a:(lambda:0,[],[]))
    monkeypatch.setattr(preflight.starts,'start',lambda case:'source')
    monkeypatch.setattr(preflight.starts,'load_native',lambda path:lambda:0)
    monkeypatch.setattr(preflight,'calibrate',lambda *args:{'threshold':1.7},raising=False)
    calls = []
    def check(a,b,protocol,count,threshold):
        calls.append((a is b,count,threshold))
        return {'threshold':threshold,'confirmations':2,'trials':[]}
    monkeypatch.setattr(preflight,'check_noise',check)
    result = preflight.run({'public_id':4},'calibrate',tmp_path,
                           {'timing':{'improvement_threshold':1.01},'preflight_seeds':[7713,8823]})
    assert calls==[(True,20,1.01)] and result['confirmations']==2
