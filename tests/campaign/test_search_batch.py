from pathlib import Path
import os
import subprocess
import tarfile

import pytest

from kernel_study.campaign import jobs
from kernel_study.tasks.kernelbench import experiment


def test_batch_runs_searches_for_one_case():
    plan = experiment()
    assert jobs.batch_case(plan,'multi-smoke',[0])==4
    assert jobs.batch_case(plan,'row-qualification',[2,3])==40
    for indices in ([],[-1],[0,2],[0,0],[4]):
        with pytest.raises(ValueError):
            jobs.batch_case(plan,'row-qualification',indices)


def test_batch_keeps_credentials_within_the_search_process(tmp_path):
    root = Path(__file__).resolve().parents[2]
    scripts = tmp_path/'project/scripts'
    scripts.mkdir(parents=True)
    (scripts/'ls6_stage_runtime.sh').write_text('cd "$work/project"\n')
    (tmp_path/'.openai_env').write_text('OPENAI_API_KEY=test-only\n')
    binary = tmp_path/'python'
    binary.write_text('#!/bin/bash\necho "${OPENAI_API_KEY:+key}|$*" >> "$TRACE"\n'
                      'if [[ $* == *"import batch_case"* ]]; then echo 4; fi\n'
                      'if [[ $* == *--start-deadline* ]]; then exit "$GUARD_RESULT"; fi\n')
    binary.chmod(0o700)
    control = tmp_path/'scontrol'
    control.write_text('#!/bin/bash\necho "|${1} ${2}" >> "$TRACE"\n')
    control.chmod(0o700)
    trace = tmp_path/'calls'
    env = dict(os.environ, BASH_ENV=os.devnull, PATH=str(tmp_path)+':'+os.environ['PATH'], HOME=str(tmp_path),
               SLURM_SUBMIT_DIR=str(tmp_path),SLURM_JOB_ID='123',TRACE=str(trace),OPENAI_API_KEY='inherited')
    for status,runs,requeues in [(0,1,0),(75,0,1),(1,0,0)]:
        trace.write_text('')
        result=subprocess.run(['bash',root/'scripts/ls6_search.sbatch','multi-smoke','0'],env=dict(env,GUARD_RESULT=str(status)))
        assert result.returncode==(1 if status==1 else 0)
        calls=trace.read_text().splitlines()
        assert any('--start-deadline' in line and '--case 4' in line for line in calls)
        assert sum(line.startswith('key|') for line in calls)==runs
        assert sum(' run multi-smoke --job 0 ' in line for line in calls)==runs
        assert sum('--auto-retry-transport' in line for line in calls)==runs
        assert calls.count('|requeue 123')==requeues


def test_node_runtime_loads_the_submitted_shallow_checkout(tmp_path):
    root = Path(__file__).resolve().parents[2]
    work, tools, node = (tmp_path/name for name in ('work','bin','node'))
    for path in (work,tools,node,tmp_path/'venv/bin',tmp_path/'upstream/KernelAgent'):
        path.mkdir(parents=True)
    original = tmp_path/'original'
    subprocess.run(['git','init','-q',original],check=True)
    for value in ('old','new'):
        (original/'version').write_text(value)
        subprocess.run(['git','-C',original,'add','version'],check=True)
        subprocess.run(['git','-C',original,'-c','user.name=test','-c','user.email=test@localhost',
                        'commit','-qm',value],check=True)
    subprocess.run(['git','clone','-q','--depth=1',original.as_uri(),work/'project'],check=True)
    expected=subprocess.check_output(['git','-C',work/'project','rev-parse','HEAD'],text=True).strip()
    (work/'cuda-module').write_text('cuda/test')
    (work/'runtime.bytes').write_text('10000\n')
    (tmp_path/'venv/bin/python').write_text('#!/bin/sh\nexit 0\n')
    (tmp_path/'venv/bin/python').chmod(0o700)
    with tarfile.open(work/'runtime.tar.zst','w') as archive:
        for name in ('venv','upstream'):
            archive.add(tmp_path/name,arcname=name)
    for name,body in [('findmnt','echo xfs'),('zstd','cat')]:
        (tools/name).write_text('#!/bin/sh\n'+body+'\n')
        (tools/name).chmod(0o700)
    env=dict(os.environ,BASH_ENV=os.devnull,PATH=str(tools)+':'+os.environ['PATH'],SLURM_TMPDIR=str(node),
             SLURM_JOB_ID='123',work=str(work),guard=str(work/'guard'))
    command='''set -eu
module(){ :; }
source "$1"
test "$(cat version)" = new
for path in "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH"; do
    [[ $path == "$node_dir/"* ]]
done
git rev-parse HEAD'''
    proc=subprocess.run(['bash','-c',command,'bash',root/'scripts/ls6_stage_runtime.sh'],env=env,text=True,capture_output=True)
    assert proc.returncode==0,proc.stderr
    assert proc.stdout.strip()==expected
    recorded=(work/'guard/runtime.txt').read_text().splitlines()
    assert recorded[0]==expected and all(Path(p).is_relative_to(node) for p in recorded[1:4])
    assert not any(node.iterdir())
