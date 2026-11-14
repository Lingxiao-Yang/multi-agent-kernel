from __future__ import annotations
import argparse
import json
from pathlib import Path
from .core import ROOT, VARIANTS, cases, environment, read_json, write_json


def main():
    p = argparse.ArgumentParser(description='Budget-matched kernel optimization study')
    sub = p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor')
    manifest = sub.add_parser('manifest')
    manifest.add_argument('--output',default='configs/primary_manifest.json')
    pilot = sub.add_parser('pilot')
    pilot.add_argument('--output',required=True)
    pilot.add_argument('--case',default='all')
    pilot.add_argument('--quick',action='store_true')
    run = sub.add_parser('run')
    run.add_argument('--case',required=True)
    run.add_argument('--variant',choices=VARIANTS,required=True)
    run.add_argument('--seed',type=int,default=11)
    run.add_argument('--output',required=True)
    run.add_argument('--provider',choices=['scripted','llm'],default='scripted')
    run.add_argument('--protocol',default=str(ROOT/'configs/protocol.json'))
    run.add_argument('--attempts',type=int)
    run.add_argument('--model')
    run.add_argument('--endpoint')
    run.add_argument('--dollar-cap',type=float)
    run.add_argument('--input-price',type=float)
    run.add_argument('--output-price',type=float)
    run.add_argument('--drop-role',choices=['analyze','advise','review'])
    run.add_argument('--drop-category')
    report = sub.add_parser('analyze')
    report.add_argument('--input',required=True)
    report.add_argument('--output',required=True)
    report.add_argument('--allow-controls',action='store_true')
    args = p.parse_args()
    if args.command == 'doctor':
        print(json.dumps(environment(),indent=2)); return
    if args.command == 'manifest':
        jobs = [{'case':c['id'],'variant':v,'seed':s} for c in cases() for s in [11,22,33] for v in VARIANTS]
        # Randomized job order limits order effects; all variants remain balanced.
        import random
        random.Random(20261114).shuffle(jobs)
        write_json(args.output,jobs)
        print(f'Wrote {len(jobs)} balanced searches'); return
    if args.command == 'analyze':
        from .analysis import analyze
        print(json.dumps(analyze(args.input,args.output,args.allow_controls),indent=2)); return
    if args.command == 'pilot':
        from .pilot import pilot
        pilot(args.output,args.case,args.quick); return
    from .providers import ChatProvider, ScriptedProvider
    from .search import search
    protocol = read_json(args.protocol)
    if args.attempts is not None:
        if args.attempts <= 0:
            p.error('--attempts must be positive')
        protocol['candidate_cap'] = args.attempts
    provider = ScriptedProvider()
    if args.provider == 'llm':
        protocol['model'] = args.model or protocol['model']
        protocol['endpoint'] = args.endpoint or protocol['endpoint']
        provider = ChatProvider(protocol['model'],protocol['endpoint'],protocol['max_output_tokens'],protocol['temperature'],
                                input_price=args.input_price,output_price=args.output_price,dollar_cap=args.dollar_cap)
    selected = next((c for c in cases() if c['id']==args.case),None)
    if selected is None:
        p.error('unknown benchmark case')
    summary = search(selected,args.variant,args.seed,protocol,provider,args.output,args.drop_role,args.drop_category)
    print(json.dumps({k:v for k,v in summary.items() if k in ['status','speedup','budget','wall_seconds','stop_reason']},indent=2))


if __name__ == '__main__':
    main()
