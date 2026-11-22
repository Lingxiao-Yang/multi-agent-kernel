import pytest
import os
from pathlib import Path

from kernel_study import bench
from kernel_study.core import read_json


def test_public_requests_reach_same_referee(monkeypatch):
    from kernel_study import public_cases
    seen = []
    monkeypatch.setattr(public_cases, 'evaluate', lambda r: seen.append(r) or {'status':'ok'})
    for interface in ('native','declarative'):
        request = {'case':{'public_id':4},'interface':interface}
        assert bench.evaluate(request)['status'] == 'ok'
    assert [r['interface'] for r in seen] == ['native','declarative']


def test_model_budget_spans_analysis_proposal_and_failed_repair(tmp_path):
    from kernel_study.reference_search import Ledger
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
    from kernel_study.reference_search import Ledger
    ledger = Ledger(tmp_path, 'sample', 4, 16, 2., 8., 0.011)
    ledger.reserve('proposal', 1000, 1000)
    reopened = Ledger(tmp_path, 'sample', 4, 16, 2., 8., 0.011)
    with pytest.raises(RuntimeError,match='dollar_cap'):
        reopened.reserve('analysis', 1000, 1000)
    assert reopened.summary()['calls'] == 1
    assert reopened.summary()['committed_usd'] == pytest.approx(0.010)


def test_both_model_interfaces_meter_the_wire_payload(tmp_path, monkeypatch):
    import io
    import json
    import urllib.request
    from kernel_study.reference_search import Ledger, MeteredChat
    sent = []
    def response(request, timeout):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({'model':'pinned','choices':[{'message':{'content':'{}'}}],
            'usage':{'prompt_tokens':20,'completion_tokens':7}}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',response)
    ledger = Ledger(tmp_path,'pair',2,2,1.,1.,1.)
    provider = MeteredChat(ledger,tmp_path,model='pinned',endpoint='http://localhost:1',
                          max_output_tokens=100,input_price=1.,output_price=1.)
    provider.complete('single',{},'analyze',11)
    messages = [{'role':'user','content':'native bottleneck diagnosis'}]
    provider.send_messages(messages,'proposal',12,max_tokens=25)
    assert sent[0]['response_format']=={'type':'json_object'}
    assert 'response_format' not in sent[1] and sent[1]['messages']==messages
    assert sent[1]['max_completion_tokens']==25
    with pytest.raises(RuntimeError): provider.send_messages(messages,'repair',13)
    assert len(sent)==2 and ledger.summary()['input_tokens']==40


def test_campaign_selects_only_registered_jobs_and_separate_calibration():
    from kernel_study.agent_campaign import jobs
    from kernel_study.public_cases import experiment
    plan = experiment()
    assert len(jobs(plan,'development')) == 12
    assert len(jobs(plan,'holdout')) == 18
    assert jobs(plan,'smoke') == [(4, 'greedy', 771)]
    assert jobs(plan,'calibration') == [(4, 'greedy', 772)]
    for i in range(0,12,2):
        pair = jobs(plan,'development')[i:i+2]
        assert {p[1] for p in pair} == {'single','greedy'}
        assert len({(p[0],p[2]) for p in pair}) == 1


def test_native_adapter_uses_upstream_parser_and_greedy_manager():
    pytest.importorskip('triton_kernel_agent')
    from kernel_study.agent_campaign import native_manager
    from triton_kernel_agent.opt_manager import OptimizationManager
    assert native_manager() is OptimizationManager


@pytest.mark.skipif(not os.getenv('KERNEL_BASELINE_PREFLIGHT'), reason='requires frozen GPU preflight')
def test_common_worker_with_official_matvec_and_hidden_seed(tmp_path):
    from kernel_study.core import ROOT
    from kernel_study.public_cases import experiment
    from kernel_study.search import run_worker
    plan = experiment()
    case = plan['cases']['4']
    base = Path(os.environ['KERNEL_BASELINE_PREFLIGHT'])/'4'
    config = read_json(base/'preflight.json')['launch']
    protocol = dict(read_json(ROOT/'configs/protocol.json'),**{'final_validation_seeds':[10001]})
    protocol.update(plan['timing'],worker_timeout_s=600)
    for interface,source in [('native','input.py'),('declarative','device.py')]:
        result = run_worker(dict(case=case,protocol=protocol,config=config,source=str(base/source),
                                 interface=interface,final=True,library=True),tmp_path,interface)
        assert result['status']=='ok', result.get('error')
        assert result['validation'][0]['inputs_unchanged']
        assert all(len(result[k]['samples_ms'])==9 for k in ('library_candidate','library','compile_candidate','compile'))


@pytest.mark.skipif(not os.getenv('KERNEL_BASELINE_DENIED_CONTROL'), reason='requires the counter-denied GPU')
def test_native_greedy_on_counter_denied_device(tmp_path):
    from kernel_study.agent_campaign import run_native
    from kernel_study.core import ROOT
    from kernel_study.public_cases import experiment
    from kernel_study.reference_search import Ledger, MeteredChat
    plan = experiment()
    base = Path(os.environ['KERNEL_BASELINE_PREFLIGHT'])/'4'
    usage = Ledger(tmp_path,'counter-control',1,16,2.,8.,0.)
    output = tmp_path/'native'
    provider = MeteredChat(usage,output,model='offline-control',endpoint='http://localhost:1',
                           input_price=2.,output_price=8.)
    protocol = read_json(ROOT/'configs/protocol.json')
    protocol.update(plan['timing'],validation_seeds=[101],final_validation_seeds=[10001],worker_timeout_s=600)
    result = run_native(plan['cases']['4'],771,protocol,provider,output,base,1)
    assert result['budget']['calls']==0 and result['status']=='interrupted'
    assert result['final']['candidate_id']==0
    assert 'No module named' not in (output/'greedy/manager.log').read_text()
