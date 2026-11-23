import ast
from math import comb
import json
import os
from pathlib import Path
import subprocess

import pytest

from kernel_study.judge.policy import check_source
from kernel_study.tasks.kernelbench import experiment
from kernel_study.tasks.starts import start
from scripts import sweep_rows as sweep


@pytest.mark.parametrize('case_id',[23,40])
def test_row_grid_changes_launches_but_keeps_the_three_kernels(case_id):
    case=experiment()['cases'][str(case_id)]
    original=start(case)
    assert sweep.variant(case,original,sweep.BASELINE)==original
    source=sweep.variant(case,original,(512,4096,2,8,3))
    check_source(source)
    before,after=(ast.parse(text).body for text in [original,source])
    assert [ast.dump(n) for n in before[:-1]]==[ast.dump(n) for n in after[:-1]]
    calls=[s.value for s in after[-1].body if isinstance(s,ast.Expr)]
    rows=case['output_shape'][0]
    width=393216 if case_id==23 else 4194304
    groups=(width+511)//512
    scratch=after[-1].body[1].value.args[0]
    assert ast.literal_eval(scratch)==(2*rows*(groups+1),)
    assert ast.literal_eval(calls[0].func.slice)==(groups,rows)
    assert ast.literal_eval(calls[1].func.slice)==(rows,)
    assert ast.literal_eval(calls[2].func.slice)==((rows*width+4095)//4096,)
    assert [ast.literal_eval(v) for v in calls[1].args[1:6]]==[rows,width,groups,512,1<<(groups-1).bit_length()]
    assert [c.keywords[0].value.value for c in calls]==[2,2,8]
    assert all(c.keywords[1].value.value==3 for c in calls)
    order=sweep.configurations([1])
    assert len(set(order))==len(order)==256 and order[0]==sweep.BASELINE
    assert len(sweep.configurations([1,2,3,4]))==1024


@pytest.mark.parametrize('scenario,levels',[('flat',[1]),('faster',[1,2,3,4]),('slower',[1,2,3,4]),
                                           ('transient',[1]),('error',[1,2,3,4])])
def test_stage_pilot_changes_only_stages_and_checks_both_directions(tmp_path,monkeypatch,scenario,levels):
    calls=[]
    def measure(output,name,case,protocol,source,incumbent,final=False):
        import re
        candidate=int(re.search('num_stages=(\\d+)',source.read_text()).group(1))
        parent=int(re.search('num_stages=(\\d+)',incumbent.read_text()).group(1))
        assert re.sub('num_stages=\\d+','num_stages=X',source.read_text())==re.sub('num_stages=\\d+','num_stages=X',incumbent.read_text())
        calls.append((candidate,parent))
        if scenario=='error' and candidate==3: return {'status':'compile_error'}
        speedup=1.
        if candidate==3 and scenario=='faster': speedup=1.05
        if candidate==3 and scenario in ('slower','transient'): speedup=.95
        if parent==3 and scenario=='slower': speedup=1.05
        return dict(status='ok',paired_speedup=speedup,confirmed=speedup>1.01)
    monkeypatch.setattr(sweep.launch_sweep,'measure',measure)
    pilot=sweep.stage_pilot(tmp_path,experiment()['cases']['23'],{'improvement_threshold':1.01})
    assert pilot['stages']==levels and len(pilot['checks'])==9
    assert calls.count((2,1))==calls.count((3,1))==calls.count((4,1))==3
    if scenario in ('slower','transient'): assert calls.count((1,3))==3


def test_four_random_configurations_are_distinct_and_failures_stay_in_the_draw():
    values=sweep.random_four([2.,3.,5.,7.,9.],9,3)
    assert values['median_ms']==3. and values['p25_ms']==2.
    assert values['probability_confirmed_gain']==pytest.approx(1-comb(6,4)/comb(9,4))
    empty=sweep.random_four([2.],8,1)
    assert empty['p75_ms'] is None and empty['probability_no_valid']==.5


def test_sweep_batch_preflights_and_calibrates_without_a_key(tmp_path):
    script=Path(__file__).resolve().parents[2]/'scripts/ls6_row_sweep.sbatch'
    archive=tmp_path/'kernel-study-campaign/a100-matvec-20261122'
    binpath=archive/'venv/bin'
    binpath.mkdir(parents=True)
    (archive/'cuda-module').write_text('cuda/test')
    (binpath/'activate').write_text('export PATH="$SCRATCH/kernel-study-campaign/a100-matvec-20261122/venv/bin:$PATH"\n')
    (tmp_path/'project').mkdir()
    trace=tmp_path/'calls'
    (binpath/'python').write_text('#!/usr/bin/env python3\nimport json,os,sys\nwith open(os.environ["TRACE"],"a") as f: print(json.dumps([sys.argv[1:],"OPENAI_API_KEY" in os.environ]),file=f)\n')
    (binpath/'python').chmod(0o700)
    env=dict(os.environ,BASH_ENV=os.devnull,SCRATCH=str(tmp_path),SLURM_SUBMIT_DIR=str(tmp_path),SLURM_JOB_ID='8',TRACE=str(trace),OPENAI_API_KEY='unused')
    for case in ['23','40']:
        subprocess.run(['bash','-c','module(){ :; }; export -f module; bash "$1" "$2"','bash',str(script),case],env=env,check=True)
    calls=[json.loads(v) for v in trace.read_text().splitlines()]
    assert not any(has_key for _,has_key in calls)
    for i,case in enumerate(['23','40']):
        batch=[a for a,_ in calls[i*5:i*5+5]]
        assert [a[a.index('preflight')+1] for a in batch[:-1]]==['preflight','stress','calibrate','hold']
        assert all(a[a.index('--case')+1]==case for a in batch)
        assert batch[-1][:2]==['-m','scripts.sweep_rows']
