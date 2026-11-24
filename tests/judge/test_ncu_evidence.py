from types import SimpleNamespace

import pytest


@pytest.mark.upstream
def test_profile_uses_upstream_selected_kernel_and_roofline(tmp_path, monkeypatch):
    from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer
    from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import KernelProfiler
    from kernel_study.judge.evaluate import ncu_evidence

    row = {'gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed':81.,
           'sm__throughput.avg.pct_of_peak_sustained_elapsed':34.}
    metrics = {'last_kernel': row}
    calls = []
    def profile(self, kernel_file, problem_file, round_num, max_retries):
        calls.append((kernel_file, problem_file, max_retries))
        return SimpleNamespace(metrics=metrics)
    monkeypatch.setattr(KernelProfiler, 'profile_kernel', profile)
    src, problem = tmp_path/'kernel.py', tmp_path/'problem.py'
    evidence = ncu_evidence(src, problem, tmp_path/'ncu')
    assert calls == [(src, problem, 1)]
    assert evidence['metrics'] == metrics
    assert evidence['roofline'] == RooflineAnalyzer().analyze(row).to_dict()
    assert evidence['kernel_selection'] == 'last'
