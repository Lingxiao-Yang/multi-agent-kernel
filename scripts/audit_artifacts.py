"""Scan intended publishable files without printing any discovered secrets."""
import re
from pathlib import Path
root=Path(__file__).resolve().parents[1]
patterns=[re.compile(r'sk-(?:proj-)?[A-Za-z0-9_-]{30,}'),
          re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----')]
bad=[]
directories=['kernel_study','configs','knowledge','tests','scripts','cluster','docs','logs','results','reports']
for directory in directories:
    for p in (root/directory).rglob('*'):
        if p.is_file() and p.suffix in ['.py','.json','.jsonl','.log','.md','.csv','.txt','.sbatch','.sh']:
            text=p.read_text(encoding='utf-8',errors='replace')
            if any(pattern.search(text) for pattern in patterns):
                bad.append(str(p.relative_to(root)))
if bad:
    raise SystemExit('Possible credentials found in: '+', '.join(bad))
print('Publishable artifact credential scan passed.')
