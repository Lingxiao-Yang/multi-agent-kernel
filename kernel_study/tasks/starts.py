"""The frozen native module used by both agents for each task."""
from pathlib import Path

from .kernelbench import load_module


def start_path(case):
    return Path(__file__).with_name('baselines') / (case['family']+'.py')


def start(case):
    return start_path(case).read_text(encoding='utf-8')


def load_native(path):
    return load_module(path).kernel_function
