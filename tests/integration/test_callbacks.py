import json

import pytest

from kernel_study.agents import kernelagent as campaign
from kernel_study.budget import Ledger
from kernel_study.campaign.report import compare
from kernel_study.judge import select as referee
from kernel_study.providers import MeteredChat
from kernel_study.tasks.kernelbench import experiment


@pytest.mark.upstream
def test_native_run_with_only_rejected_proposals_reaches_the_phase_report(tmp_path,monkeypatch):
    import io
    import logging
    import urllib.request
    from types import SimpleNamespace
    from kernel_study.tasks.starts import start

    from triton_kernel_agent.worker import VerificationWorker
    from triton_kernel_agent.opt_worker_component.orchestrator.optimization_orchestrator import OptimizationOrchestrator

    class FailedProposals:
        def __init__(self,**options):
            self.directory=options['log_dir']
            self.directory.mkdir()
            (self.directory/'manager.log').write_text('no successful workers\n')

        def run_optimization(self,**options):
            worker=VerificationWorker(0,self.directory,self.directory,openai_model='test-model')
            obj=SimpleNamespace(logger=logging.getLogger('campaign-test'),artifact_dir=self.directory,verification_worker=worker)
            generate = OptimizationOrchestrator._generate_optimized_kernel
            assert generate(obj,'make a proposal',1) is None
            rejected = generate(obj,'make a proposal',2)
            worker._write_files(rejected,[options['test_code']])
            passed,_,error = worker._run_test()
            assert not passed and 'StaticRejection' in error
            return dict(success=False,total_rounds=2,kernel_code=None)

    replies = iter(('No source this round.', start({'family':'gelu'}).replace('return Y','return torch.sin(X)')))
    def response(*args,**kwargs):
        return io.BytesIO(json.dumps({'model':'test-model','choices':[{'message':{'content':next(replies)}}],
            'usage':{'prompt_tokens':10,'completion_tokens':5}}).encode())
    monkeypatch.setattr(urllib.request,'urlopen',response)
    monkeypatch.setattr(campaign,'native_manager',lambda:FailedProposals)
    measurement=dict(status='ok',candidate={'median_ms':1.},incumbent={'median_ms':1.},
                     compile={'median_ms':1.},compile_candidate={'median_ms':1.},paired_speedup=1.)
    monkeypatch.setattr(referee,'run_worker',lambda *args:measurement)
    output=tmp_path/'greedy'
    ledger=Ledger(tmp_path,'greedy',2,2,1.,1.,1.)
    provider=MeteredChat(ledger,output,model='test-model',endpoint='http://localhost:1')
    from kernel_study.records import ROOT
    import os
    monkeypatch.setenv('PYTHONPATH',os.pathsep.join((str(ROOT),os.environ.get('PYTHONPATH',''))))
    case=dict(experiment()['cases']['4'],problem='public matvec task')
    (tmp_path/'input.py').write_text(start(case))
    result=campaign.run_native(case,11,{},provider,output,tmp_path,2)
    assert not result['upstream_success'] and result['status']=='complete'
    greedy=referee.audit(output,'greedy')
    assert greedy['fallback'] and not greedy['correct_new'] and greedy['budget']['attempts']==2
    report=compare([dict(greedy,agent='single'),greedy],[4],[11],{(4,11):measurement})
    assert report['paired_searches']==1 and report['fallbacks']['greedy']==1
    assert report['rejections']['greedy']['static_check']==1
