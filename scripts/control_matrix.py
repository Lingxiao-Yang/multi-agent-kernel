"""Four-arm actual-GPU plumbing test with identical deterministic proposals."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, VARIANTS, cases, read_json
from kernel_study.providers import ScriptedProvider
from kernel_study.search import search

protocol=read_json(ROOT/'configs/protocol.json')
protocol.update(candidate_cap=2,call_cap=8)
case=next(c for c in cases() if c['id']=='softmax_irregular')
for variant in VARIANTS:
    result=search(case,variant,11,protocol,ScriptedProvider(),ROOT/'results/gpu_control_matrix'/variant)
    print(variant,result['status'],result.get('speedup'),flush=True)
