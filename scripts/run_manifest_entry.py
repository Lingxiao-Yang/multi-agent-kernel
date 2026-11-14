import argparse
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import read_json

p=argparse.ArgumentParser()
p.add_argument('--index',type=int,required=True)
p.add_argument('--output',required=True)
args,extra=p.parse_known_args()
entry=read_json('configs/primary_manifest.json')[args.index]
dest=Path(args.output)/f"{entry['case']}_{entry['variant']}_{entry['seed']}"
if dest.exists():
    raise SystemExit('Refusing to overwrite an existing run; inspect or use a new output root.')
subprocess.run([sys.executable,'-m','kernel_study.cli','run','--provider','llm',
               '--case',entry['case'],'--variant',entry['variant'],'--seed',str(entry['seed']),
               '--output',str(dest),*extra],check=True)
