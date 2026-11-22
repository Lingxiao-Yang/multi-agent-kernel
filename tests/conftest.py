import os
from importlib.util import find_spec
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line('markers', 'gpu: requires CUDA and Triton')
    config.addinivalue_line('markers', 'upstream: requires the registered upstream checkouts')


def pytest_collection_modifyitems(items):
    gpu_items = [item for item in items if item.get_closest_marker('gpu')]
    if gpu_items:
        available = False
        if find_spec('torch') and find_spec('triton'):
            import torch
            available = torch.cuda.is_available()
        if not available:
            for item in gpu_items:
                item.add_marker(pytest.mark.skip(reason='requires CUDA and Triton'))
    if not os.getenv('KERNEL_BASELINE_UPSTREAM'):
        for item in items:
            if item.get_closest_marker('upstream'):
                item.add_marker(pytest.mark.skip(reason='requires registered upstream checkouts'))


@pytest.fixture
def public_case():
    return dict(public_id=26, id='test', family='gelu', dtype='float32',
                shape=[8, 32], output_shape=[8, 32], atol=1e-5, rtol=1e-5)


@pytest.fixture
def mock_environment(monkeypatch):
    monkeypatch.setattr('kernel_study.search.environment', lambda: {'gpu': 'test'})


@pytest.fixture
def preflight_dir():
    path = os.getenv('KERNEL_BASELINE_PREFLIGHT')
    if not path:
        pytest.skip('requires frozen GPU preflight')
    return Path(path)
