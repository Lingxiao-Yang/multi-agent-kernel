import pytest

from kernel_study.campaign.jobs import jobs, phase_budget
from kernel_study.cli import main
from kernel_study.records import read_json, write_json
from kernel_study.tasks.kernelbench import experiment


def test_qualification_runs_exactly_six_searches_with_the_same_caps():
    plan = experiment()
    assert jobs(plan,'matvec-recheck')==[(4,'single',11),(4,'single',22)]
    assert jobs(plan,'row-qualification')==[(case,agent,11) for case in (23,40) for agent in ('greedy','single')]
    for phase in ('matvec-recheck','row-qualification'):
        assert phase_budget(plan,phase)=={'rounds':4,'call_cap':16,'candidate_cap':4}


def save_audit(root,phase,case,seed,agent,ms,fallback=False):
    path=root/f'{phase}-{case}-{agent}-{seed}'
    write_json(path/'audit.json',dict(agent=agent,case=case,seed=seed,status='complete',correct_new=True,
               fallback=fallback,budget={'attempts':4},candidates=[],cost_usd=.12,initial_ms=10.,latency_ms=ms))
    if agent=='single':
        write_json(path/'matched-evidence.json',{'identical':True,'metric_count':27})


def test_matvec_recheck_uses_both_standalone_endpoints_and_charges(tmp_path):
    phase='matvec-recheck'
    save_audit(tmp_path,phase,4,11,'single',4.10)
    main(['report',phase,str(tmp_path),'--require-seed','11'])
    partial=read_json(tmp_path/(phase+'-report.json'))
    assert partial['decision']=='incomplete' and partial['seeds']['11']['passed']
    assert partial['seeds']['11']['below_3_57_ms'] is False
    assert partial['seeds']['11']['cost_usd']==.12
    save_audit(tmp_path,phase,4,22,'single',4.101)
    with pytest.raises(SystemExit) as stopped:
        main(['report',phase,str(tmp_path),'--require-seed','22'])
    result=read_json(tmp_path/(phase+'-report.json'))
    assert stopped.value.code==3 and result['decision']=='matvec-recheck-failed'
    save_audit(tmp_path,phase,4,11,'single',3.56)
    save_audit(tmp_path,phase,4,22,'single',4.09)
    main(['report',phase,str(tmp_path)])
    result=read_json(tmp_path/(phase+'-report.json'))
    assert result['decision']=='matvec-recheck-passed' and result['seeds']['11']['below_3_57_ms']


def test_row_qualification_reads_each_same_card_pair(tmp_path):
    phase='row-qualification'
    for case,ratio in [(23,.9),(40,1.03)]:
        for agent,ms in [('single',ratio),('greedy',1.)]:
            save_audit(tmp_path,phase,case,11,agent,ms)
        write_json(tmp_path/'pairs'/f'{phase}-{case}-11/direct.result.json',
                   dict(status='ok',candidate={'median_ms':ratio},incumbent={'median_ms':1.}))
    main(['report',phase,str(tmp_path)])
    result=read_json(tmp_path/(phase+'-report.json'))
    assert result['decision']=='row-qualification-failed'
    assert result['tasks']['23']['outcome']=='single-faster'
    assert result['tasks']['40']['outcome']=='single-slower'
    assert result['tasks']['23']['reference_ms']['best_launch']==pytest.approx(13.507584)
    for agent in ('single','greedy'):
        save_audit(tmp_path,phase,40,11,agent,1.,fallback=True)
    main(['report',phase,str(tmp_path)])
    result=read_json(tmp_path/(phase+'-report.json'))
    assert result['decision']=='row-qualification-passed'
    assert result['tasks']['40']['outcome']=='both-retained-start'
    (tmp_path/f'{phase}-23-single-11/matched-evidence.json').unlink()
    main(['report',phase,str(tmp_path)])
    assert read_json(tmp_path/(phase+'-report.json'))['decision']=='incomplete'


def test_slurm_submits_only_the_two_row_pairs_once_after_a_partial_retry(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    bindir=tmp_path/'bin'
    bindir.mkdir()
    sbatch=bindir/'sbatch'
    sbatch.write_text('#!/bin/bash\necho "Site accounting active"\n'
                      'if [[ ${!#} == layernorm && ! -f "$CALLS.retry" ]]; then touch "$CALLS.retry"; exit 2; fi\n'
                      'echo "$*" >> "$CALLS"\nwc -l < "$CALLS"\n')
    sbatch.chmod(0o700)
    calls=tmp_path/'calls'
    env=dict(os.environ,PATH=str(bindir)+':'+os.environ['PATH'],CALLS=str(calls))
    script=Path(__file__).resolve().parents[2]/'scripts/ls6_submit_qualification.sh'
    assert subprocess.run(['bash',str(script)],cwd=tmp_path,env=env).returncode!=0
    for _ in range(2):
        subprocess.run(['bash',str(script)],cwd=tmp_path,env=env,check=True)
    submitted=calls.read_text().splitlines()
    assert [line.split()[-1] for line in submitted]==['softmax','layernorm']
    assert all('--dependency' not in line for line in submitted)
