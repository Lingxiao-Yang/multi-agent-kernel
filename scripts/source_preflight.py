"""Validate source-v1 baseline ABI on GPU without inference or optimized code."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, cases, read_json, write_json
from kernel_study.source_space import baseline
from kernel_study.search import run_worker

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--cases',default='')
args=p.parse_args()
out=Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
protocol=read_json(ROOT/'configs/protocol.json')
records=[]
for case in cases():
    if args.cases and case['id'] not in args.cases.split(','): continue
    source,config=baseline(case)
    path=out/(case['id']+'.py'); path.write_text(source,encoding='utf-8')
    result=run_worker(dict(case=case,config=config,source=str(path),protocol=protocol,final=True,library=True,profile=True),out,case['id'])
    records.append(dict(case=case['id'],result=result))
    write_json(out/'preflight.json',records)
    print(case['id'],result['status'],result.get('error',''),flush=True)
    if result['status']!='ok': sys.exit(1)
