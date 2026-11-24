from pathlib import Path
import subprocess

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
    import os
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
        assert calls.count('|requeue 123')==requeues
