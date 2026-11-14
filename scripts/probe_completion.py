"""One minimal, bounded diagnostic request. No private project content."""
import getpass
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.providers import ChatProvider
from kernel_study.core import write_json
os.environ['OPENAI_API_KEY']=getpass.getpass('API key for one diagnostic request (not saved): ')
p=ChatProvider('gpt-4.1-mini-2025-04-14','https://api.openai.com/v1',16,.7,input_price=.4,output_price=1.6,dollar_cap=.01)
try:
    r=p.complete('Return JSON only.',{'instruction':'Return {"ok":true}.'},'diagnostic',11)
    result={'status':'ok','model':r.model,'input_tokens':r.input_tokens,'output_tokens':r.output_tokens,'reply':r.text,'estimated_usd':p.completed_cost}
except Exception as exc:
    result={'status':'error','error':str(exc),'estimated_completed_usd':p.completed_cost,'budget_committed_usd':p.spent}
write_json('logs/completion_access.json',result)
print(result)
