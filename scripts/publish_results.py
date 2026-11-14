"""Copy result artifacts to the separate results worktree; never commit here.

Explicit roots avoid credentials, environments, caches, and the nested worktree.
Use after experiments finish, then inspect/audit and commit from that worktree.
"""
import shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1]
target=root/'tmp/results-worktree'
if not (target/'.git').exists():
    raise SystemExit('Create the results worktree first')
for name in ['results','reports','logs']:
    if (root/name).exists():
        shutil.copytree(root/name,target/name,dirs_exist_ok=True)
print('Copied results/reports/logs to',target)
