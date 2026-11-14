import pytest
from kernel_study.metrics import throughput, work_and_traffic


def test_matmul_counts_and_units():
    case={'family':'matmul','shape':[10,20,30],'dtype':'float16'}
    assert work_and_traffic(case)=={'flops':12000,'minimum_bytes':2200}
    r=throughput(case,1,peak_gbps=10,peak_tflops=1)
    assert r['logical_gbps']==pytest.approx(.0022)
    assert r['achieved_tflops']==pytest.approx(.000012)


def test_no_unverified_roofline_fraction():
    r=throughput({'family':'softmax','shape':[10,20],'dtype':'float32'},1)
    assert r['minimum_bytes']==1600
    assert r['fraction_peak_bandwidth'] is None
    assert r['fraction_peak_compute'] is None
