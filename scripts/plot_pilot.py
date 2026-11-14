import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kernel_study.core import read_json, write_json

source=Path(sys.argv[1]); output=Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True)
rows=read_json(source/'pilot_results.json')
table=[]
for row in rows:
    r=row['result']
    if r['status']!='ok':
        table.append({'case':row['case'],'label':row['label'],'status':r['status']})
        continue
    latency=r['candidate']['median_ms']
    table.append({'case':row['case'],'label':row['label'],'status':'ok','median_ms':latency,
                  'iqr_ms':r['candidate']['p75_ms']-r['candidate']['p25_ms'],
                  'paired_speedup':r['paired_speedup'],'confirmed':r['confirmed'],
                  'library_ms':r['library']['median_ms'],
                  'max_abs_error':max(x['max_abs_error'] for x in r['validation'])})
keys=sorted(set().union(*(r.keys() for r in table)))
with (output/'pilot.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(table)
transformed=[r for r in table if r['label']=='transformed' and r['status']=='ok']
fig,ax=plt.subplots(figsize=(10,5))
ax.barh([r['case'] for r in transformed],[r['paired_speedup'] for r in transformed],color='#356f94')
ax.axvline(1,color='black',linewidth=1)
ax.set(xlabel='Paired speedup of predefined transformation',title='GPU engineering pilot — not an LLM comparison')
ax.set_xscale('log'); ax.grid(axis='x',alpha=.2)
fig.tight_layout(); fig.savefig(output/'pilot_speedups.png',dpi=180); plt.close(fig)
write_json(output/'validation_summary.json',{'evaluations':len(rows),'passed':sum(r['status']=='ok' for r in table),
    'failed':sum(r['status']!='ok' for r in table),'hardware':read_json(source/'manifest.json')['environment'],
    'warning':'Predefined transforms, laptop GPU, small sample pilot. Do not infer LLM or A100 effects.'})
print('Wrote',output)
