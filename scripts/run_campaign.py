"""Portable balanced real-model campaign with a single shared spending cap."""
import argparse
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, VARIANTS, cases, read_json, write_json
from kernel_study.providers import ChatProvider
from kernel_study.search import search

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--attempts',type=int,default=2)
p.add_argument('--cases',default='softmax_irregular,matmul_rectangular')
p.add_argument('--seeds',default='11,22,33')
p.add_argument('--dollar-cap',type=float,default=5.)
p.add_argument('--model',default='gpt-4.1-mini-2025-04-14')
p.add_argument('--endpoint',default='https://api.openai.com/v1')
p.add_argument('--input-price',type=float,default=.4)
p.add_argument('--output-price',type=float,default=1.6)
p.add_argument('--validated-pilot',help='Require 24 successful GPU engineering evaluations from this directory')
args=p.parse_args()
if not 1<=args.attempts<=20 or args.dollar_cap<=0:
    p.error('attempts must be 1..20 and dollar cap positive')
chosen=args.cases.split(','); seeds=[int(s) for s in args.seeds.split(',')]
if set(chosen)-{c['id'] for c in cases()}:
    p.error('unknown case')
if args.validated_pilot:
    measured=read_json(Path(args.validated_pilot)/'pilot_results.json')
    if len(measured)!=24 or any(row['result']['status']!='ok' for row in measured):
        p.error('The prerequisite GPU pilot is incomplete or contains validation failures')
output=Path(args.output).resolve(); output.mkdir(parents=True,exist_ok=False)
protocol=read_json(ROOT/'configs/protocol.json')
protocol.update(candidate_cap=args.attempts,call_cap=4*args.attempts,model=args.model,endpoint=args.endpoint,
                input_usd_per_million=args.input_price,output_usd_per_million=args.output_price)
provider=ChatProvider(args.model,args.endpoint,protocol['max_output_tokens'],protocol['temperature'],
                      input_price=args.input_price,output_price=args.output_price,dollar_cap=args.dollar_cap)
jobs=[(c,v,s) for c in cases() if c['id'] in chosen for s in seeds for v in VARIANTS]
random.Random(20261114).shuffle(jobs)
write_json(output/'pilot_protocol.json',protocol)
write_json(output/'campaign_manifest.json',{'jobs':[{'case':c['id'],'variant':v,'seed':s} for c,v,s in jobs],
    'shared_dollar_cap':args.dollar_cap,'phase':'pilot','note':'Reduced pilot caps, not the full proposed primary study.'})
completed=0
for case,variant,seed in jobs:
    result=search(case,variant,seed,protocol,provider,output/f"{case['id']}_{variant}_{seed}")
    completed+=result['status']=='complete'
    print(case['id'],variant,seed,result['status'],result.get('speedup'),provider.spent,flush=True)
    write_json(output/'cost.json',{'estimated_completed_usd':provider.completed_cost,
        'budget_committed_usd':provider.spent,'cap_usd':args.dollar_cap,'completed_searches':completed})
    if result['status']!='complete':
        print('Campaign stopped:',result.get('stop_reason',result['status']),flush=True)
        break
if completed==len(jobs):
    from kernel_study.analysis import analyze
    analyze(output,output/'analysis')
