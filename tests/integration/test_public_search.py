import os

import pytest

from kernel_study.agents.kernelagent import run_native
from kernel_study.budget import Ledger
from kernel_study.judge.evaluate import run_worker
from kernel_study.providers import MeteredChat
from kernel_study.records import ROOT, read_json
from kernel_study.tasks.kernelbench import experiment


@pytest.mark.gpu
@pytest.mark.upstream
def test_common_worker_with_official_matvec_and_hidden_seed(tmp_path,preflight_dir):
    plan = experiment()
    case = plan['cases']['4']
    base = preflight_dir/'4'
    protocol = dict(read_json(ROOT/'configs/protocol.json'), **{'final_validation_seeds':[10001]})
    protocol.update(plan['timing'],worker_timeout_s=600)
    result = run_worker(dict(case=case,protocol=protocol,source=str(base/'input.py'),
                             interface='native',final=True,library=True,profile=True),tmp_path,'native')
    assert result['status']=='ok', result.get('error')
    assert result['validation'][0]['inputs_unchanged']
    assert result['profile']['kernels'] and result['profile']['kernels'][0]['registers_per_thread']
    assert all(len(result[k]['samples_ms'])==9 for k in ('library_candidate','library','compile_candidate','compile'))


@pytest.mark.gpu
@pytest.mark.upstream
@pytest.mark.skipif(not os.getenv('KERNEL_BASELINE_DENIED_CONTROL'), reason='requires the counter-denied GPU')
def test_native_greedy_on_counter_denied_device(tmp_path,preflight_dir):
    plan = experiment()
    base = preflight_dir/'4'
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
