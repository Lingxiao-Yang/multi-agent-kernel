from collections import Counter

from scripts import measure_matvec_finalists as study


def test_remeasure_pairs_both_native_programs_with_all_references_and_each_other(monkeypatch):
    programs = {'start':lambda:1,'greedy':lambda:2}
    libraries = {'eager':lambda:3,'compile':lambda:4}
    calls=[]
    def measure(a,b,protocol):
        calls.append((a(),b()))
        return [{'median_ms':a()},{'median_ms':b()}]
    monkeypatch.setattr(study,'time_pair',measure)
    rows=study.measure(programs,libraries,{'timing_repeats':9})
    assert Counter(calls)=={pair:3 for pair in [(1,1),(1,3),(1,4),(2,2),(2,3),(2,4),(2,1)]}
    assert {r['repeat'] for r in rows}=={0,1,2}
    assert all(r['speedup']==.5 for r in rows if r['reference']=='start')
