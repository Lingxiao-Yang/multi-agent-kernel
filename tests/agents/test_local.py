import copy
import json
from pathlib import Path

import pytest

from kernel_study.agents import local
from kernel_study.judge import select
from kernel_study.providers import Reply
from kernel_study.records import ROOT, code_hashes, read_json


class SourceProvider:
    name = 'test'
    model = 'test-model'
    source = """import triton
import triton.language as tl

@triton.jit
def gelu(X, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X+i, i<TOTAL, other=0.)
    tl.store(Y+i, x, i<TOTAL)
"""

    def complete(self, system, context, stage, seed):
        answer = dict(config=context['incumbent']['config'], source=self.source) if stage=='implement' else {}
        return Reply(json.dumps(answer), 1, 1, self.model, {})


@pytest.fixture
def protocol():
    return dict(read_json(ROOT/'configs/protocol.json'), search_space='source-v1', candidate_cap=2, call_cap=8)


def mock_worker(monkeypatch, worker):
    monkeypatch.setattr(local, 'run_worker', worker)
    monkeypatch.setattr(select, 'run_worker', worker)


def measurement(confirmed=False):
    return dict(status='ok', confirmed=confirmed, paired_speedup=2. if confirmed else 1.,
                candidate={'median_ms':1.}, incumbent={'median_ms':2. if confirmed else 1.})


def test_api_failure_is_not_completed_scientific_search(tmp_path, monkeypatch, public_case, mock_environment, protocol):
    class QuotaRejectedProvider(SourceProvider):
        name = 'llm'

        def complete(self, *args):
            raise RuntimeError('HTTP 429: credit_balance_exhausted')

    mock_worker(monkeypatch, lambda *args: measurement())
    result = local.search(public_case,'single',11,protocol,QuotaRejectedProvider(),tmp_path/'run')
    assert result['status']=='interrupted' and result['stop_reason'].startswith('HTTP 429')
    assert not result['scientific_evidence']
    assert result['budget']['calls']==1 and result['budget']['tokens']==0
    assert result['estimated_inference_usd']==0


def test_incorrect_candidates_keep_baseline_and_count_calls(tmp_path, monkeypatch, public_case, mock_environment, protocol):
    def worker(request, directory, label):
        if label.startswith('candidate_'):
            assert Path(request['source']).read_text()==SourceProvider.source
            return {'status':'incorrect','error':'injected wrong output'}
        return measurement()

    mock_worker(monkeypatch, worker)
    result = local.search(public_case,'multi_knowledge',11,protocol,SourceProvider(),tmp_path/'run')
    assert result['manifest']['code_hashes'] == code_hashes()
    assert result['budget']['attempts']==2 and result['budget']['calls']==8
    assert result['final']['candidate_id']==0 and result['speedup']==1
    assert not result['scientific_evidence']
    assert [row['result']['status'] for row in result['history']]==['incorrect','incorrect']
    assert all(not row['accepted'] for row in result['history'])


def test_fresh_final_validation_falls_back(tmp_path, monkeypatch, public_case, mock_environment, protocol):
    checks = []

    def worker(request, directory, label):
        candidate = Path(request['source']).name=='candidate_001.py'
        if request.get('final'):
            checks.append(label)
            if label=='final_001':
                return {'status':'incorrect','error':'held-out failure'}
        return measurement(confirmed=candidate)

    mock_worker(monkeypatch, worker)
    protocol['candidate_cap'] = 1
    result = local.search(public_case,'single',11,protocol,SourceProvider(),tmp_path/'run')
    assert result['history'][0]['accepted']
    assert checks==['audit_001','final_001','final_000']
    assert result['final']['candidate_id']==0 and result['speedup']==1


def test_source_search_rejects_null_source_and_counts_failed_attempts(tmp_path, monkeypatch, public_case, mock_environment, protocol):
    contexts = []

    class NullSourceProvider(SourceProvider):
        source = None

        def complete(self, system, context, stage, seed):
            assert 'matmul_tensor' not in context['incumbent']['source']
            contexts.append((stage,copy.deepcopy(context)))
            return super().complete(system, context, stage, seed)

    workers = []

    def worker(request, directory, label):
        workers.append(label)
        return measurement()

    mock_worker(monkeypatch, worker)
    result = local.search(public_case,'multi',11,protocol,NullSourceProvider(),tmp_path/'run')
    assert result['budget']['attempts']==2 and result['budget']['calls']==8
    assert workers==['baseline','final_000']
    assert all('source_edit_required' in x['result']['error'] for x in result['history'])
    assert result['speedup']==1 and result['final']['candidate_id']==0
    assert contexts[-1][1]['proposed_candidate']['source'] is None
    assert 'previous_proposal' in contexts[4][1]
