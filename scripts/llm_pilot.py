"""Small balanced real-model pilot; one shared $5 cap across all runs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, VARIANTS, cases, read_json, write_json
from kernel_study.providers import ChatProvider
from kernel_study.search import search

# Serialize against the engineering pilot on the same physical GPU.
import time
pilot_file=ROOT/'results/local_pilot_v2/pilot_results.json'
while not pilot_file.exists() or len(read_json(pilot_file)) < 24 or not (ROOT/'tmp/gpu_ready_for_llm').exists():
    print('Waiting for engineering pilot to release the GPU...',flush=True)
    time.sleep(10)

protocol=read_json(ROOT/'configs/protocol.json')
protocol.update(candidate_cap=2,call_cap=8,max_output_tokens=4096,
                model='gpt-4.1-mini-2025-04-14',endpoint='https://api.openai.com/v1',
                input_usd_per_million=.4,output_usd_per_million=1.6)
provider=ChatProvider(protocol['model'],protocol['endpoint'],4096,.7,
                      input_price=.4,output_price=1.6,dollar_cap=5.)
output=ROOT/'results/llm_local_pilot'
output.mkdir(parents=True,exist_ok=False)
write_json(output/'pilot_protocol.json',protocol)
jobs=[(c,v,s) for c in cases() if c['id'] in ['softmax_irregular','matmul_rectangular']
      for s in [11,22,33] for v in VARIANTS]
import random
random.Random(20261114).shuffle(jobs)
for case,variant,seed in jobs:
    before=provider.spent
    result=search(case,variant,seed,protocol,provider,output/f"{case['id']}_{variant}_{seed}")
    print(case['id'],variant,seed,result['status'],result.get('speedup'),provider.spent,flush=True)
    write_json(output/'cost.json',{'estimated_total_usd':provider.spent,'cap_usd':5.})
    if result.get('stop_reason') and any(x in result['stop_reason'] for x in ['HTTP','dollar_cap','missing_usage']):
        break
