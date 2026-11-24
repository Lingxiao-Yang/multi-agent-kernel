import json
import os
from pathlib import Path
import subprocess
import tarfile

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
if 'EXPECT_NODE_RUNTIME' in os.environ:
    node = Path(os.environ['SLURM_TMPDIR'])
    assert Path(sys.argv[0]).is_relative_to(node) and Path.cwd().is_relative_to(node)
    assert Path('revision.txt').read_text()=='registered'
    upstream = Path(os.environ['KERNEL_BASELINE_UPSTREAM'])
    assert upstream.is_relative_to(node) and (upstream/'KernelAgent').is_dir()
    assert os.environ['PYTHONPATH'].split(os.pathsep)==[str(Path.cwd()),str(upstream/'KernelAgent')]
if '--start-deadline' in args: sys.exit(int(os.environ['GUARD']))
"""
    for name in ['python','scontrol']:
        (bindir/name).write_text(stub)
        (bindir/name).chmod(0o700)
    env=dict(os.environ,BASH_ENV=os.devnull,SCRATCH=str(tmp_path),SLURM_SUBMIT_DIR=str(tmp_path),HOME=str(tmp_path),
             SLURM_JOB_ID='123',TRACE=str(trace),GUARD=str(guard),OPENAI_API_KEY='inherited-test')
    if script == 'ls6_qualification.sbatch':
        project = tmp_path/'project'
        def git(*args):
            subprocess.run(['git','-C',str(project),'-c','user.name=test','-c','user.email=test@localhost',*args],check=True)
        git('init','-q')
        marker = project/'revision.txt'
        marker.write_text('archived')
        git('add','revision.txt')
        git('commit','-qm','archived')
        (archive/'upstream/KernelAgent').mkdir(parents=True)
        with tarfile.open(tmp_path/'runtime.tar.zst', 'w') as bundle:
            for name in ('venv','upstream'):
                bundle.add(archive/name,arcname=name)
            bundle.add(project,arcname='project')
        marker.write_text('registered')
        git('commit','-qam','registered')
        (tmp_path/'runtime.bytes').write_text('100000\n')
        node, tools = tmp_path/'node', tmp_path/'tools'
        node.mkdir()
        tools.mkdir()
        for name, body in [('findmnt','echo xfs'), ('zstd','cat')]:
            binary = tools/name
            binary.write_text('#!/bin/sh\n'+body+'\n')
            binary.chmod(0o700)
        env.update(SLURM_TMPDIR=str(node),EXPECT_NODE_RUNTIME='1',PATH=str(tools)+':'+env['PATH'])
    script = Path(__file__).resolve().parents[2]/'scripts'/script
    proc=subprocess.run(['bash','-c','module(){ :; }; export -f module; bash "$1" "$2"','bash',str(script),str(argument)],env=env)
    if script.name=='ls6_qualification.sbatch':
        runtime = (tmp_path/'guard-123-0/runtime.txt').read_text().splitlines()
        assert runtime[0]==subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
        assert all(Path(p).is_relative_to(node) for p in runtime[1:4])
        assert not any(node.iterdir())
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


@pytest.mark.parametrize('guard', [75,1])
def test_staged_row_stops_before_model_access_when_the_guard_fails(tmp_path,guard):
    status,calls=run_stub_batch(tmp_path,'ls6_qualification.sbatch','softmax',guard)
    assert status==(0 if guard==75 else 1)
    assert all(not key for _,_,key in calls)
    assert [name for name,_,_ in calls]==(['python','scontrol'] if guard==75 else ['python'])
