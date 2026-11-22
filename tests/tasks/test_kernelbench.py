from kernel_study.tasks.kernelbench import experiment


def test_public_tasks_and_limits_are_stated_before_search():
    plan = experiment()
    assert plan['development'] == [4, 23, 40]
    assert plan['holdout'] == [2, 26, 36]
    assert (plan['phases']['development']['candidate_cap'], plan['phases']['development']['call_cap']) == (4, 16)
    assert plan['cases']['40']['output_shape'] == [16, 64, 256, 256]
    assert plan['cases']['36']['output_shape'] == [112, 64, 512, 512]
    assert plan['cases']['23']['atol'] < 1/393216
