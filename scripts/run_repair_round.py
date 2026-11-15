"""Execute one predeclared phase, with durable decisions and capped spending."""
import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, event, read_json, write_json
from kernel_study.repair_analysis import summarize

p=argparse.ArgumentParser()
p.add_argument('--phase',choices=['mini','strong'],required=True)
p.add_argument('--output',required=True)
p.add_argument('--mini-output')
args=p.parse_args()
plan=read_json(ROOT/'configs/repair_round.json')
ceiling=plan['prior_completed_usd']+plan['local_smoke_cap_usd']+sum(plan[x]['cap_usd'] for x in ['mini','strong'])
if ceiling>plan['approved_total_usd']: raise SystemExit('Round caps exceed authorized total')
out=Path(args.output).resolve()
out.mkdir(parents=True,exist_ok=False)
log=out/'steps.jsonl'
event(log,'phase_start',phase=args.phase,plan=plan,maximum_total_exposure_usd=ceiling)
write_json(out/'plan.json',plan)
if args.phase=='strong':
    if not args.mini_output: raise SystemExit('Strong phase requires completed mini output')
    mini=Path(args.mini_output)
    if read_json(mini/'plan.json')!=plan: raise SystemExit('Plan drift between phases')
    decision=summarize(mini/'campaign',out/'mini_decision_recomputed.json',plan)
    if not decision['complete']: raise SystemExit('Mini phase is incomplete; refusing selective comparison')
    if not decision['stronger_required']:
        event(log,'phase_skipped',reason='Mini met the predeclared readiness threshold for both cases')
        write_json(out/'status.json',dict(status='skipped',reason='mini_ready'))
        raise SystemExit(0)
    event(log,'stronger_triggered',improved_searches_by_case=decision['improved_searches_by_case'])
settings=plan[args.phase]
command=[sys.executable,str(ROOT/'scripts/run_campaign.py'),'--output',str(out/'campaign'),
         '--search-space','source-v1','--shared-api-guidance','--attempts',str(plan['attempts']),
         '--cases',','.join(plan['cases']),'--seeds',','.join(map(str,plan['seeds'])),
         '--model',settings['model'],'--endpoint',plan['endpoint'],'--input-price',str(settings['input_price']),
         '--output-price',str(settings['output_price']),'--dollar-cap',str(settings['cap_usd'])]
event(log,'campaign_launch',command=command,phase_ceiling_usd=settings['cap_usd'])
result=subprocess.run(command,cwd=ROOT)
event(log,'campaign_exit',returncode=result.returncode)
report=summarize(out/'campaign',out/'repair_report.json',plan)
event(log,'phase_evaluated',complete=report['complete'],improved_searches_by_case=report['improved_searches_by_case'],
      stronger_required=report['stronger_required'],first_correct_after_attempt_2=report['first_correct_after_attempt_2'])
write_json(out/'status.json',dict(status='complete' if report['complete'] and result.returncode==0 else 'incomplete'))
raise SystemExit(0 if report['complete'] and result.returncode==0 else 2)
