import pytest
from kernel_study.core import write_json


@pytest.mark.parametrize('interface', ['native','declarative'])
def test_final_selection_checks_every_candidate_against_the_current_winner(tmp_path, monkeypatch, interface):
    from kernel_study import public_referee as referee
    start = {'id':0, 'source':str(tmp_path/'baseline.py'), 'config':None}
    candidates = []
    for i in range(5):
        path = tmp_path / ('baseline.py' if i == 0 else f'program_{i}.py')
        path.write_text(f'def kernel(x):\n    return x + {i}\n')
        if i: candidates.append(dict(id=i, source=str(path), config=None))
    trials = []
    def worker(request, directory, label):
        i = next(item['id'] for item in [start,*candidates] if item['source'] == request['source'])
        trials.append((i, request['incumbent_source'], request.get('final'), request['interface']))
        if i == 3: return {'status':'incorrect'}
        return {'status':'ok', 'confirmed':i in (2,3), 'candidate':{'median_ms':float(i or 5)}}
    monkeypatch.setattr(referee, 'run_worker', worker)
    final, checks = referee.select_final(tmp_path, {}, {}, interface, start, candidates)
    assert final['candidate_id'] == 2
    assert [c['attempt'] for c in checks] == [1,2,3,4]
    assert [c['selected'] for c in checks] == [False,True,False,False]
    assert all(t[2] and t[3] == interface for t in trials)
    assert trials[2][1] == candidates[1]['source']
    assert trials[3][1] == candidates[1]['source']


def test_pair_worker_receives_both_recorded_finalists_and_reuses_raw_samples(tmp_path, monkeypatch):
    from kernel_study import public_referee as referee
    directories = {}
    for agent, interface in [('single','declarative'),('greedy','native')]:
        directory = tmp_path/agent
        directory.mkdir()
        (directory/'winner.py').write_text('pass')
        write_json(directory/'summary.json', {'manifest':{'case':{'public_id':4},'protocol':{'p':1},'seed':11}})
        write_json(directory/'audit.json', dict(status='complete', finalist=dict(source='winner.py',config=None,interface=interface)))
        directories[agent] = directory
    calls = []
    def worker(req, where, label):
        calls.append(req)
        result = dict(status='ok',candidate={'median_ms':1.,'samples_ms':[1.]},incumbent={'median_ms':2.,'samples_ms':[2.]})
        write_json(where/(label+'.result.json'), result)
        return result
    monkeypatch.setattr(referee,'run_worker',worker)
    for _ in range(2):
        result = referee.measure_pair(tmp_path/'pair', directories['single'], directories['greedy'])
        assert result['incumbent']['samples_ms'] == [2.]
    assert len(calls) == 1
    req = calls[0]
    assert req['source'] == str(directories['single']/'winner.py')
    assert req['incumbent_source'] == str(directories['greedy']/'winner.py')
    assert (req['interface'],req['incumbent_interface']) == ('declarative','native')


def test_formatting_and_docstrings_do_not_turn_a_start_into_a_new_candidate():
    from kernel_study.reference_report import changed_program
    start = 'def run(x):\n    """baseline"""\n    return x + 1\n'
    copied = '# same launch\ndef run(x):\n    """new explanation"""\n    return x+1\n'
    assert not changed_program(copied,start)
    assert changed_program(copied.replace('x+1','x+2'),start)
