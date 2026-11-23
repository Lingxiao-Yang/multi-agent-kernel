import sys
from types import SimpleNamespace

from kernel_study.judge import timing
from kernel_study.tasks.kernelbench import experiment


def test_public_warmup_measures_gpu_work_after_compilation_and_keeps_slow_samples(monkeypatch):
    clock = [0.]
    calls = []
    counts = dict(a=0,b=0)

    class Event:
        def __init__(self,enable_timing): pass
        def record(self): self.stamp = clock[0]
        def synchronize(self): pass
        def elapsed_time(self,end): return end.stamp-self.stamp

    def forward(key):
        counts[key] += 1
        calls.append(key)
        if counts[key]==1:
            clock[0] += 1000
        elif key=='a':
            clock[0] += 5
        else:
            clock[0] += 9 if counts[key]==6 else 3

    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(Event=Event,synchronize=lambda:None)))
    a,b = timing.time_pair(lambda:forward('a'),lambda:forward('b'),
                           dict(warmup_ms=24,timing_repeats=3,inner_repeats=1,slow_sample_ratio=1.05))
    assert ''.join(calls) == 'ab'+'ab'*3+'abbaab'
    assert a['warmup_gpu_ms']==b['warmup_gpu_ms']==24
    assert a['warmup_median_ms']==5 and b['warmup_median_ms']==3
    assert a['samples_ms']==[5,5,5] and b['samples_ms']==[3,9,3]
    assert a['slow_sample_indices']==[] and b['slow_sample_indices']==[1]


def test_public_timing_uses_the_registered_recovery_duration():
    timing = experiment()['timing']
    assert timing['warmup_ms']==2610
    assert timing['slow_sample_ratio']==1.05
    assert timing['improvement_threshold']==1.01
