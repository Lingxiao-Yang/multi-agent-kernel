from collections import Counter
import importlib.util
from pathlib import Path


def test_each_idle_and_comparator_is_repeated_five_times_in_balanced_blocks():
    path = Path(__file__).resolve().parents[2]/'scripts/measure_matvec_idle.py'
    spec = importlib.util.spec_from_file_location('idle_study', path)
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
    schedule = list(study.schedule())
    expected = {(program, reference, idle) for program in ('start', 'greedy')
                for reference in ('self', 'eager', 'compile') for idle in (0, .5, 2, 5)}
    assert len(schedule) == 120
    assert Counter((p, r, idle) for _, p, r, idle in schedule) == {v:5 for v in expected}
    for repeat in range(5):
        assert {row[1:] for row in schedule if row[0] == repeat} == expected
    assert schedule == list(study.schedule())
