from kernel_study.tasks.kernelbench import experiment


def test_matvec_checks_candidate_source_before_import_and_generates_inputs_on_gpu(tmp_path):
    import subprocess
    import sys
    from kernel_study.tasks.kernelbench import test_source

    plan = experiment()
    assert plan['validation_seeds'] == [101]
    for case in plan['cases'].values():
        script = test_source(case)
        assert 'task_inputs' in script and 'check_forward' in script
        (tmp_path/'test.py').write_text(script)
        (tmp_path/'kernel.py').write_text("raise RuntimeError('candidate executed')\n")
        import os
        from kernel_study.records import ROOT
        result = subprocess.run([sys.executable, str(tmp_path/'test.py')], cwd=tmp_path,
                                env=dict(os.environ, PYTHONPATH=str(ROOT)), capture_output=True, text=True)
        assert 'StaticRejection' in result.stderr and 'RuntimeError: candidate executed' not in result.stderr


def test_public_tasks_and_limits_are_stated_before_search():
    plan = experiment()
    assert plan['development'] == [4, 23, 40]
    assert plan['holdout'] == [2, 26, 36]
    assert (plan['phases']['development']['candidate_cap'], plan['phases']['development']['call_cap']) == (4, 16)
    assert plan['cases']['40']['output_shape'] == [16, 64, 256, 256]
    assert plan['cases']['36']['output_shape'] == [112, 64, 512, 512]
    assert plan['cases']['23']['atol'] < 1/393216
