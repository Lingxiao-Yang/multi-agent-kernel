"""Read-only API availability check; never prints credentials."""
import json
import os
import urllib.request
import sys
import getpass
from pathlib import Path

key=getpass.getpass('API key (not saved): ') if '--prompt-key' in sys.argv else os.environ['OPENAI_API_KEY']
request=urllib.request.Request('https://api.openai.com/v1/models',headers={'Authorization':'Bearer '+key})
try:
    with urllib.request.urlopen(request,timeout=30) as response:
        ids=[row['id'] for row in json.load(response)['data']]
    result={'status':'ok','available_candidates':[x for x in ids if x.startswith(('gpt-4.1-mini','gpt-5-mini','gpt-5.4-mini'))]}
except Exception as exc:
    result={'status':'error','error_type':type(exc).__name__,'error':str(exc)}
Path('logs/model_access.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
