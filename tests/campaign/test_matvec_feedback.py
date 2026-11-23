from kernel_study.tasks import starts
from scripts import check_matvec_feedback as check


def test_startup_only_requeues_slow_verification():
    assert check.deadline_status(dict(seconds=20.0,passed=True,stderr=''))==0
    assert check.deadline_status(dict(seconds=20.01,passed=True,stderr=''))==75
    assert check.deadline_status(dict(seconds=20.01,passed=False,stderr='Test execution timed out after 20 seconds'))==75
    assert check.deadline_status(dict(seconds=10.,passed=False,stderr='incorrect output'))==1


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
