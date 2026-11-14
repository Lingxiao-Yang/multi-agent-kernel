"""Generate an evidence index from on-disk artifacts, never inferred outcomes."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, read_json, write_json

rows=[]
for directory in sorted((ROOT/'results').iterdir()):
    if not directory.is_dir():
        continue
    if (directory/'pilot_results.json').exists():
        data=read_json(directory/'pilot_results.json')
        rows.append({'cohort':directory.name,'type':'predefined-transform pilot',
                     'evaluations':len(data),'correct':sum(x['result']['status']=='ok' for x in data),
                     'incorrect_or_error':sum(x['result']['status']!='ok' for x in data)})
    elif (directory/'pilot_protocol.json').exists():
        runs=[read_json(p) for p in directory.rglob('summary.json')]
        completed=[r for r in runs if r.get('stop_reason') in ['candidate_cap','call_cap','token_safety_cap'] and r.get('resolved_model')]
        rows.append({'cohort':directory.name,'type':'LLM pilot','terminated_searches':len(runs),'completed_searches':len(completed),
                     'attempts':sum(x['budget']['attempts'] for x in runs),
                     'model_calls':sum(x['budget']['calls'] for x in runs),
                     'input_tokens':sum(x['budget']['input_tokens'] for x in runs),
                     'output_tokens':sum(x['budget']['output_tokens'] for x in runs),
                     'status_counts':{status:sum(x['status']==status for x in runs) for status in set(x['status'] for x in runs)}})
write_json(ROOT/'reports/evidence_index.json',rows)
print(json.dumps(rows,indent=2))
