import pytest

from kernel_study.budget import Budget, Ledger


def test_candidate_failure_and_repair_consume_budget():
    b = Budget(candidate_cap=2)
    b.reserve_attempt()  # failed compile
    b.reserve_attempt()  # attempted repair
    with pytest.raises(RuntimeError,match='candidate_cap'):
        b.reserve_attempt()


def test_calls_and_actual_tokens_are_charged_separately():
    b = Budget(call_cap=2,token_safety_cap=100)
    b.reserve_call(); b.charge_tokens(70,31)
    assert b.stop_reason() == 'token_safety_cap'
    assert b.tokens == 101  # actual usage is retained, never truncated to cap
    with pytest.raises(RuntimeError):
        b.reserve_call()


def test_call_cap_blocks_further_network_requests():
    b = Budget(call_cap=1)
    b.reserve_call()
    with pytest.raises(RuntimeError):
        b.reserve_call()


def test_local_search_cap_takes_precedence_over_an_unsent_charge(tmp_path):
    ledger=Ledger(tmp_path,'capped',1,1,1.,1.,.002)
    call=ledger.reserve('proposal',1000,1000)
    ledger.settle(call,1000,1000)
    with pytest.raises(RuntimeError,match='candidate_cap'):
        ledger.reserve('repair',1000,1000)
    assert ledger.summary()['committed_usd']==pytest.approx(.002)


def test_model_budget_spans_analysis_proposal_and_failed_repair(tmp_path):
    ledger = Ledger(tmp_path, 'matvec-11', 2, 3, 2.0, 8.0, 0.08)
    analysis = ledger.reserve('analysis', 400, 1000)
    ledger.settle(analysis,300,300)
    candidate = ledger.reserve('proposal', 500, 1000)
    ledger.settle(candidate,400,400)
    repair = ledger.reserve('repair', 500, 1000)
    ledger.settle(repair,100,50)
    with pytest.raises(RuntimeError):
        ledger.reserve('repair', 500, 1000)
    summary = ledger.summary()
    assert (summary['attempts'], summary['calls']) == (2,3)
    assert summary['input_tokens'] == 800 and summary['output_tokens'] == 750
    assert summary['completed_usd'] == pytest.approx(0.0076)
    next_job = Ledger(tmp_path,'next',4,16,2.,8.,0.08)
    with pytest.raises(RuntimeError,match='dollar_cap'):
        next_job.reserve('analysis', 40000, 1000)
    assert next_job.summary()['calls'] == 0


def test_ambiguous_call_stays_reserved_across_process_restarts(tmp_path):
    ledger = Ledger(tmp_path, 'sample', 4, 16, 2., 8., 0.011)
    ledger.reserve('proposal', 1000, 1000)
    reopened = Ledger(tmp_path, 'sample', 4, 16, 2., 8., 0.011)
    with pytest.raises(RuntimeError,match='dollar_cap'):
        reopened.reserve('analysis', 1000, 1000)
    assert reopened.summary()['calls'] == 1
    assert reopened.summary()['committed_usd'] == pytest.approx(0.010)
