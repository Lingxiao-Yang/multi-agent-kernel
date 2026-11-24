import ast

import pytest

from kernel_study.records import digest
from kernel_study.tasks import kernelbench, starts


HASHES = {
    '2':'f9e41154bea9556175f0c45dc10d6ae1e7ad1e4ed9e479d95b266ce283e0b857',
    '4':'6f4d7a7578e55dd40c5f966cbe8ae9c2317e389fb7a3f15fbfacf7c70ef4a165',
    '23':'1b1ff1e691cacad6b077e955c89154ab8a014412b5c866cbca590da3626fe55e',
    '26':'ea89595c75bf44ad1668eae6eb765b3bd5d587c141bf03dd7e6baf95bf69a499',
    '36':'a5caed22cf12b17aa2232b834ab72d9149511675299a752d6a4c0664d757f103',
    '40':'37f7937591f684ab13199d22ff4609861afdb5f7e7c41df23bac39e80ccf079d',
}


@pytest.mark.parametrize('number', HASHES)
def test_frozen_native_start_matches_measured_module(number):
    source = starts.start(kernelbench.experiment()['cases'][number])
    assert digest(source) == HASHES[number]
    names = [n.name for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)]
    assert names.count('kernel_function') == 1


@pytest.mark.upstream
def test_matvec_matches_upstream_example():
    assert starts.start(kernelbench.experiment()['cases']['4']) == kernelbench.matvec_path().read_text()
