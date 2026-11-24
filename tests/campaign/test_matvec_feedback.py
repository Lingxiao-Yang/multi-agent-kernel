from types import SimpleNamespace
import sys

import pytest

from kernel_study.records import read_json
from kernel_study.tasks import starts
from scripts import check_matvec_feedback as check


def test_startup_only_requeues_slow_verification():
    assert check.deadline_status(dict(seconds=25.0,passed=True,stderr=''))==0
    assert check.deadline_status(dict(seconds=25.01,passed=True,stderr=''))==75
    assert check.deadline_status(dict(seconds=30.2,passed=False,stderr='Test execution timed out after 30 seconds'))==75
    assert check.deadline_status(dict(seconds=26.,passed=False,stderr='incorrect output'))==1


@pytest.mark.parametrize('runs,status', [
    ([(True,25,'')], 0),
    ([(False,30.2,'Test execution timed out after 30 seconds'),(True,26,''),(True,23,'')], 0),
    ([(True,26,'')]*3, 75),
    ([(True,26,''),(False,27,'incorrect output')], 1),
])
def test_startup_retries_only_slow_checks_in_the_same_allocation(tmp_path, monkeypatch, runs, status):
    worker = SimpleNamespace(workdir=tmp_path, test_timeout_s=30)
    responses = iter(runs)
    def verify(actual, source, case):
        assert actual is worker and actual.test_timeout_s == 30
        passed, seconds, stderr = next(responses)
        return dict(passed=passed, seconds=seconds, timeout_s=30, stderr=stderr)
    monkeypatch.setattr(check, 'verify', verify)
    assert check.check_start(worker, 'source', {}, {'torch':'test'}) == status
    record = read_json(tmp_path/'verification.json')
    assert [row['seconds'] for row in record['attempts']] == [seconds for _,seconds,_ in runs]
    assert record['passed'] == runs[-1][0] and record['torch'] == 'test'


@pytest.mark.parametrize('case_id', ['2','4','23','26','36','40'])
def test_startup_entry_runs_each_frozen_case_with_the_upstream_timeout(tmp_path, monkeypatch, case_id):
    class Worker:
        def __init__(self, worker_id, workdir, log_dir, test_timeout_s=30):
            assert workdir == log_dir == tmp_path and test_timeout_s == 30
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(__version__='torch'))
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(__version__='triton'))
    monkeypatch.setitem(sys.modules, 'triton_kernel_agent.worker', SimpleNamespace(VerificationWorker=Worker))
    monkeypatch.setattr(check.kernelbench, 'task_path', lambda case: tmp_path/'problem.py')
    monkeypatch.setattr(check.kernelbench, 'load_module', lambda path: None)
    def start(worker, source, case, versions):
        assert str(case['public_id']) == case_id and source == starts.start_path(case)
        assert versions == {'torch':'torch', 'triton':'triton'}
        return 0
    monkeypatch.setattr(check, 'check_start', start, raising=False)
    monkeypatch.setattr(sys, 'argv', ['check', str(tmp_path), '--start-deadline', '--case', case_id])
    with pytest.raises(SystemExit) as result:
        check.main()
    assert result.value.code == 0


def test_verification_uses_the_upstream_deadline_and_keeps_failures(tmp_path):
    class Worker:
        test_timeout_s = 30

        def _write_files(self, kernel, tests):
            assert kernel == starts.start({'family':'matvec'})
            assert len(tests) == 1 and 'check_source' in tests[0]

        def _run_test(self):
            return False, '', 'test output'

    row = check.verify(Worker(), starts.start_path({'family':'matvec'}), {'family':'matvec','public_id':4})
    assert row['passed'] is False and row['timeout_s']==30 and row['stderr']=='test output'
    assert row['seconds'] >= 0
