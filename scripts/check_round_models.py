"""Read-only check for the exact pinned models; never log credentials/catalog."""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT,read_json,write_json,utcnow
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
plan=read_json(ROOT/'configs/repair_round.json')
needed=[plan[x]['model'] for x in ['mini','strong']]
request=urllib.request.Request(plan['endpoint']+'/models',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']})
try:
    with urllib.request.urlopen(request,timeout=30) as response:
        found={item['id'] for item in json.load(response)['data']}
    available={name:name in found for name in needed}
    result=dict(checked_utc=utcnow(),available=available,status='ok' if all(available.values()) else 'missing_snapshot',inference_calls=0)
except Exception as exc:
    result=dict(checked_utc=utcnow(),status='error',error_type=type(exc).__name__,inference_calls=0)
write_json(a.output,result);print(json.dumps(result))
raise SystemExit(0 if result['status']=='ok' else 2)
