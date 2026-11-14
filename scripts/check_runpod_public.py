"""Read-only public capacity probe; never creates resources or uses credentials."""
import json
import urllib.request
from pathlib import Path
query='query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice secure: lowestPrice(input: {gpuCount: 1, secureCloud: true}) {stockStatus uninterruptablePrice availableGpuCounts} community: lowestPrice(input: {gpuCount: 1, secureCloud: false}) {stockStatus uninterruptablePrice availableGpuCounts} } }'
req=urllib.request.Request('https://api.runpod.io/graphql',data=json.dumps({'query':query}).encode(),headers={'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=30) as response:
        data=json.load(response)
    if 'errors' in data:
        result={'status':'query_error','errors':data['errors']}
    else:
        result={'status':'ok','a100_options':[x for x in data['data']['gpuTypes'] if 'A100' in x['id']]}
except Exception as exc:
    result={'status':'unavailable','error':str(exc)}
Path('reports/runpod_public_capacity.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
