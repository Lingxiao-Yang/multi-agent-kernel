import json
import os
from pathlib import Path
import subprocess

import pytest


def run_stub_batch(tmp_path, script, argument, guard):
    archive = tmp_path/'kernel-study-campaign/a100-matvec-20261122'
    bindir = archive/'venv/bin'
    bindir.mkdir(parents=True)
    (tmp_path/'project').mkdir()
    (tmp_path/'campaign').mkdir()
    (archive/'cuda-module').write_text('cuda/test')
    (bindir/'activate').write_text('export PATH="$SCRATCH/kernel-study-campaign/a100-matvec-20261122/venv/bin:$PATH"\n')
    (tmp_path/'.openai_env').write_text('OPENAI_API_KEY=test-only\n')
    trace = tmp_path/'calls'
    stub = """#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
args=sys.argv[1:]
with open(os.environ['TRACE'],'a') as f:
    print(json.dumps([Path(sys.argv[0]).name,args,'OPENAI_API_KEY' in os.environ]),file=f)
if '--start-deadline' in args: sys.exit(int(os.environ['GUARD']))
"""
    for name in ['python','scontrol']:
        (bindir/name).write_text(stub)
        (bindir/name).chmod(0o700)
    env=dict(os.environ,BASH_ENV=os.devnull,SCRATCH=str(tmp_path),SLURM_SUBMIT_DIR=str(tmp_path),HOME=str(tmp_path),
             SLURM_JOB_ID='123',TRACE=str(trace),GUARD=str(guard),OPENAI_API_KEY='inherited-test')
    script = Path(__file__).resolve().parents[2]/'scripts'/script
    proc=subprocess.run(['bash','-c','module(){ :; }; export -f module; bash "$1" "$2"','bash',str(script),str(argument)],env=env)
    return proc.returncode,[json.loads(line) for line in trace.read_text().splitlines()]


@pytest.mark.parametrize('guard,first,searches,requeues',[(0,0,['0','1'],0),(0,2,['2','3'],0),(75,0,[],1),(1,2,[],0)])
def test_pair_only_starts_after_guard(tmp_path,guard,first,searches,requeues):
    status,calls=run_stub_batch(tmp_path,'ls6_matvec_development.sbatch',first,guard)
    assert status==(1 if guard==1 else 0)
    assert '--start-deadline' in calls[0][1] and not calls[0][2]
    assert [a[a.index('--job')+1] for name,a,key in calls if 'run' in a and key]==searches
    assert [a for name,a,key in calls if name=='scontrol' and not key]==[['requeue','123']]*requeues
    assert all(('run' in a)==key for _,a,key in calls)


@pytest.mark.parametrize('name,case,phase,indices',[
    ('matvec-11','4','matvec-recheck',['0']),('matvec-22','4','matvec-recheck',['1']),
    ('softmax','23','row-qualification',['0','1']),('layernorm','40','row-qualification',['2','3'])])
def test_qualification_batch_runs_only_assigned_searches(tmp_path,name,case,phase,indices):
    status,calls=run_stub_batch(tmp_path,'ls6_qualification.sbatch',name,0)
    assert status==0 and calls[0][1][calls[0][1].index('--case')+1]==case
    searches=[a for _,a,key in calls if key]
    assert all('run' in a and phase in a for a in searches)
    assert [a[a.index('--job')+1] for a in searches]==indices
    _,report,key=calls[-1]
    assert not key and 'report' in report and phase in report
    if phase=='matvec-recheck':
        assert report[report.index('--require-seed')+1]==name.split('-')[1]
