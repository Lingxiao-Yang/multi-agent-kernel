import copy
import errno
import json
import sqlite3
from pathlib import Path

import pytest

from kernel_study.agents import local
from kernel_study.budget import Ledger
from kernel_study.judge import select
from kernel_study.providers import ChatProvider, MeteredChat, Reply
from kernel_study.records import ROOT, code_hashes, read_json


class SourceProvider:
    name = 'test'
    model = 'test-model'
    source = """import torch
import triton
import triton.language as tl

@triton.jit
def gelu(X, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X+i, i<TOTAL, other=0.)
    tl.store(Y+i, x, i<TOTAL)

def kernel_function(X):
    Y = torch.empty_like(X)
    gelu[(1,)](X, Y, X.numel(), 256)
    return Y
"""

    def complete(self, system, context, stage, seed):
        answer = dict(source=self.source, rationale='tune the program') if stage=='implement' else {}
        return Reply(json.dumps(answer), 1, 1, self.model, {})


@pytest.fixture
def protocol():
    return dict(read_json(ROOT/'configs/protocol.json'), search_space='native', candidate_cap=2, call_cap=8)


def test_single_receives_shared_files_and_accepts_a_launch_only_change(tmp_path, monkeypatch, public_case, mock_environment, protocol):
    from kernel_study.tasks.starts import start
    case = dict(public_case,problem='pytorch task',test='correctness script')
    baseline = start(case)
    class NativeProvider(SourceProvider):
        def complete(self, system, context, stage, seed):
            assert context['incumbent']['source'] == baseline
            assert 'starting_module' not in context
            assert context['hardware']['specs'] == local.kernelbench.experiment()['gpu_specs']
            assert context['case']['problem'] == case['problem'] and context['case']['test'] == case['test']
            assert 'launch_contract' not in context and 'config' not in context['incumbent']
            self.source = baseline.replace('num_warps=4','num_warps=2')
            return super().complete(system, context, stage, seed)
    def worker(request, directory, label):
        assert request['interface']=='native' and 'config' not in request
        return measurement(confirmed='candidate_001' in request['source'])
    mock_worker(monkeypatch, worker)
    result = local.search(case,'single',11,dict(protocol,candidate_cap=1),NativeProvider(),tmp_path/'run')
    assert result['final']['candidate_id']==1 and result['final']['interface']=='native'


def mock_worker(monkeypatch, worker):
    monkeypatch.setattr(local, 'run_worker', worker)
    monkeypatch.setattr(select, 'run_worker', worker)


@pytest.mark.parametrize('variant',['single','multi','single_knowledge','multi_knowledge'])
def test_only_the_implementation_stage_supplies_code(tmp_path,monkeypatch,public_case,mock_environment,protocol,variant):
    prompts = {}
    class Provider(SourceProvider):
        def complete(self, system, context, stage, seed):
            prompts[stage] = system
            if stage!='implement':
                assert all('source' not in context.get(k,{}) for k in ('analyze','advise'))
                return Reply(json.dumps({stage:'text','source':'unexpected module'}),1,1,self.model,{})
            return super().complete(system,context,stage,seed)
    mock_worker(monkeypatch,lambda *args: measurement())
    result=local.search(public_case,variant,11,dict(protocol,candidate_cap=1),Provider(),tmp_path/'run')
    for stage in ('analyze','advise','review'):
        assert 'no source code' in prompts[stage].lower() and 'complete Python module' not in prompts[stage]
    assert 'three short sentences' in prompts['advise'] and 'plain language' in prompts['advise']
    assert 'complete Python module' in prompts['implement'] and 'untested' in prompts['implement']
    assert 'source' not in result['history'][0]['review']


@pytest.mark.parametrize('variant',['single','multi'])
def test_repeated_incumbent_is_charged_but_never_timed_and_search_continues(tmp_path,monkeypatch,public_case,mock_environment,protocol,variant):
    workers, contexts = [], []
    class Provider(SourceProvider):
        def complete(self,system,context,stage,seed):
            if stage=='analyze':
                contexts.append(copy.deepcopy(context))
            if stage=='implement':
                attempt=context['attempt']
                source = [self.source,self.source+'\n# same program\n',self.source.replace('256','128')][attempt-1]
                return Reply(json.dumps(dict(source=source,rationale='test a change')),1,1,self.model,{})
            return super().complete(system,context,stage,seed)
    def worker(request,directory,label):
        workers.append(label)
        return measurement(confirmed=True)
    mock_worker(monkeypatch,worker)
    out=tmp_path/'run'
    result=local.search(public_case,variant,11,dict(protocol,candidate_cap=3,call_cap=12),Provider(),out)
    repeated=contexts[2]['history'][-1]['result']
    assert repeated['status']=='unchanged_incumbent' and 'no new measurement' in repeated['reason'].lower()
    assert 'candidate' not in repeated
    assert not {'candidate_002','profile_002','audit_002','final_002'} & set(workers)
    assert contexts[2]['incumbent']['id']==1 and result['final']['candidate_id']==3
    assert result['budget']['attempts']==3 and result['budget']['calls']==12
    assert (out/'candidate_002.py').is_file() and not result['history'][1]['accepted']


def test_matched_smoke_shares_only_initial_counter_sample(tmp_path,monkeypatch,public_case,mock_environment,protocol):
    metrics={'start':{'counter':12}}
    profiled=[]
    def worker(request,directory,label):
        if request.get('profile'):
            profiled.append((label,request.get('shared_ncu_metrics')))
        return measurement(confirmed=label.startswith('candidate_'))
    mock_worker(monkeypatch,worker)
    local.search(public_case,'single',771,dict(protocol,candidate_cap=1,shared_ncu_metrics=metrics),
                 SourceProvider(),tmp_path/'run')
    assert profiled==[('baseline',metrics),('profile_001',None)]


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


@pytest.mark.parametrize('failure', ['sqlite', 'settlement', 'proposal_file'])
def test_storage_or_accounting_failure_interrupts_search(tmp_path, monkeypatch, public_case, mock_environment, protocol, failure):
    out = tmp_path/'run'
    ledger = Ledger(tmp_path, out.name, 2, 8, 2., 8., 1.)
    provider = MeteredChat(ledger,out,model='test-model',endpoint='http://localhost:1')
    stages = []

    def send(self, messages, stage, *args):
        stages.append(stage)
        return SourceProvider().complete(None, None, stage, None)

    if failure == 'sqlite':
        execute = sqlite3.connect

        def fail_connect(*args, **kwargs):
            if not stages:
                raise sqlite3.OperationalError('disk I/O error')
            return execute(*args, **kwargs)

        monkeypatch.setattr(sqlite3, 'connect', fail_connect)
        calls, reason = [], 'disk I/O error'
    elif failure == 'settlement':
        settle = ledger.settle
        monkeypatch.setattr(ledger, 'settle', lambda number,i,o: settle(number+1,i,o))
        calls, reason = ['analyze'], 'unreserved or already settled request'
    else:
        write_json = local.write_json

        def fail_proposal(path, value):
            if path.name == 'proposal_001.json':
                raise OSError(errno.EIO, 'proposal file I/O error')
            write_json(path, value)

        monkeypatch.setattr(local, 'write_json', fail_proposal)
        calls, reason = ['analyze','advise','implement'], 'proposal file I/O error'
    monkeypatch.setattr(ChatProvider,'send_messages',send)
    mock_worker(monkeypatch,lambda *args: measurement())
    result = local.search(public_case,'single',11,protocol,provider,out)
    assert result['status']=='interrupted' and reason in result['stop_reason']
    assert stages==calls and result['history']==[] and result['final'] is None
    assert read_json(out/'summary.json')['stop_reason']==result['stop_reason']


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
    assert result['selection'][0]['endpoint_status']=='incorrect'
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


def test_native_submission_requires_a_rationale(tmp_path, monkeypatch, public_case, mock_environment, protocol):
    class NoRationale(SourceProvider):
        def complete(self, system, context, stage, seed):
            response = {'source':self.source} if stage=='implement' else {}
            return Reply(json.dumps(response), 1, 1, self.model, {})
    workers = []
    def worker(request, directory, label):
        workers.append(label)
        return measurement()
    mock_worker(monkeypatch, worker)
    result = local.search(public_case,'single',11,dict(protocol,candidate_cap=1),NoRationale(),tmp_path/'run')
    assert workers == ['baseline','final_000']
    assert 'rationale' in result['history'][0]['result']['error']
