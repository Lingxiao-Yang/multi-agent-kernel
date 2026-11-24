from argparse import ArgumentParser, Namespace

import pytest

from kernel_study.campaign import jobs as campaign
from kernel_study.cli import main
from kernel_study.records import digest,read_json,write_json


def registration(root, knowledge=False):
    import subprocess
    from kernel_study.records import ROOT
    revision=subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip()
    write_json(root/'role-study.json',dict(revision=revision,cases=[4,23],seeds=[44,55],knowledge=knowledge))


def test_role_schedule_counterbalances_order_and_uses_fresh_seeds(tmp_path):
    registration(tmp_path)
    plan=campaign.campaign_plan('roles',tmp_path)
    schedule=campaign.jobs(plan,'roles')
    assert schedule[:4]==[(4,'single',44),(4,'multi',44),(4,'multi',55),(4,'single',55)]
    assert schedule[4:6]==[(23,'multi',44),(23,'single',44)]
    assert campaign.phase_budget(plan,'roles')==dict(rounds=4,candidate_cap=4,call_cap=16)
    registration(tmp_path,True)
    schedule=campaign.jobs(campaign.campaign_plan('roles',tmp_path),'roles')
    assert len(schedule)==16 and set(schedule)=={(c,a,s) for c in (4,23) for s in (44,55)
                                              for a in ('single','multi','single_knowledge','multi_knowledge')}
    spec=read_json(tmp_path/'role-study.json')
    write_json(tmp_path/'role-study.json',dict(spec,seeds=[11,44]))
    with pytest.raises(ValueError): campaign.campaign_plan('roles',tmp_path)


def test_registered_case_ids_are_integers(tmp_path):
    registration(tmp_path)
    spec=read_json(tmp_path/'role-study.json')
    write_json(tmp_path/'role-study.json',dict(spec,cases=['4']))
    with pytest.raises(ValueError,match='case IDs'):
        campaign.campaign_plan('roles',tmp_path)


def test_role_runs_share_measured_counters_and_compare_the_actual_two_finalists(tmp_path,monkeypatch):
    from kernel_study.campaign import roles
    output=tmp_path/'campaign'
    registration(output)
    base=tmp_path/'starts/4'
    base.mkdir(parents=True)
    (base/'input.py').write_text('starter')
    problem=tmp_path/'problem.py'
    problem.write_text('task')
    identity=dict(native_source_hash=digest('starter'),problem_hash=digest('task'))
    write_json(base/'preflight.json',dict(identity,checks={'native':{}}))
    write_json(base/'stress.json',dict(identity,checks={'native':[{}]}))
    counters=tmp_path/'counters.json'
    write_json(counters,{'counter_access_verified':True})
    args=Namespace(phase='roles',job=0,preflight=base.parent,output=output,counters=counters,model='test-model',
                   endpoint='http://localhost:1',input_price=2.,output_price=8.,retry_transport=False)
    monkeypatch.setenv('OPENAI_API_KEY','test-only')
    monkeypatch.setattr(campaign,'check_device',lambda plan:dict(uuid='GPU-1',node='node1',driver=plan['driver']))
    monkeypatch.setattr(campaign.kernelbench,'upstream',lambda name:tmp_path)
    monkeypatch.setattr(campaign.kernelbench,'task_path',lambda case:problem)
    monkeypatch.setattr(campaign.kernelbench,'test_source',lambda case:'test')
    monkeypatch.setattr(campaign.starts,'start',lambda case:'starter')
    measurements,searches,pairs=[],[],[]
    raw={'kernel':{'counter':123.}}
    def profile(source,task,folder):
        assert source.read_text()=='starter' and task==problem
        measurements.append(folder)
        return {'metrics':raw}
    def search(case,agent,seed,protocol,provider,folder):
        searches.append((agent,seed,protocol['shared_ncu_metrics']))
        write_json(folder/'audit.json',{})
        return {'status':'complete'}
    def pair(folder,a,b):
        if (a/'audit.json').is_file() and (b/'audit.json').is_file():
            pairs.append((a.name,b.name))
            return {'status':'ok'}
    monkeypatch.setattr(roles,'ncu_evidence',profile)
    monkeypatch.setattr(campaign,'search',search)
    monkeypatch.setattr(campaign,'audit',lambda *a:None)
    monkeypatch.setattr(campaign,'describe_ncu',lambda x:{'metrics':x})
    monkeypatch.setattr(campaign,'check_single_evidence',lambda *a:{'identical':True})
    monkeypatch.setattr(campaign,'measure_pair',pair)
    for i in (0,1):
        args.job=i
        campaign.run_campaign(args,ArgumentParser())
    assert len(measurements)==1 and searches==[('single',44,raw),('multi',44,raw)]
    assert pairs==[('roles-4-single-44','roles-4-multi-44')]
    spec=read_json(output/'role-study.json')
    write_json(output/'role-study.json',dict(spec,revision='another'))
    with pytest.raises(SystemExit): campaign.run_campaign(args,ArgumentParser())
    assert len(searches)==2


def test_role_report_counts_searches_with_confirmed_gains_and_uses_direct_measurements(tmp_path):
    registration(tmp_path)
    for case in (4,23):
        for seed in (44,55):
            for agent in ('single','multi'):
                gained=case==4 and agent=='multi'
                path=tmp_path/f'roles-{case}-{agent}-{seed}'
                write_json(path/'audit.json',dict(agent=agent,case=case,seed=seed,status='complete',fallback=not gained,
                           initial_ms=10.,latency_ms=8. if gained else 10.,cost_usd=.12,committed_usd=.12,
                           budget={'attempts':1},candidates=[dict(attempt=1,usd=.12,status='ok',latency_ms=8.,selected=gained)]))
                write_json(path/'matched-evidence.json',dict(identical=True))
            pair=tmp_path/'pairs'/f'roles-{case}-{seed}'/'single-vs-multi'
            write_json(pair/'direct.result.json',dict(status='ok',candidate={'median_ms':4. if case==4 else 1.},incumbent={'median_ms':1.}))
    main(['report','roles',str(tmp_path)])
    r=read_json(tmp_path/'roles-report.json')
    assert r['complete'] and r['comparisons']['single-vs-multi']['ratio']==pytest.approx(2.)
    assert r['tasks']['4']['multi']['confirmed_gain_searches']==2
    assert r['tasks']['23']['single']['fallbacks']==2
    assert r['comparisons']['single-vs-multi']['matched_budget_ms']
    (tmp_path/'pairs/roles-4-44/single-vs-multi/direct.result.json').unlink()
    main(['report','roles',str(tmp_path)])
    r=read_json(tmp_path/'roles-report.json')
    assert not r['complete'] and r['comparisons']['single-vs-multi']['ratio'] is None
