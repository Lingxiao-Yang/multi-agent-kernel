"""Audit actual proposal source and measured outcomes, independently of rationale.

Static features identify code mechanisms to inspect; they do not prove speedup
causality, semantic novelty, or that a primitive dominates execution time.
"""
import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import ROOT, digest, read_json, write_json


def features(text, config):
    tree=ast.parse(text)
    functions={x.name:x for x in tree.body if isinstance(x,ast.FunctionDef)}
    names=[x['kernel'] for x in config.get('launches',[])]
    nodes=[n for name in names if name in functions for n in ast.walk(functions[name])]
    calls=Counter(ast.unparse(n.func) for n in nodes if isinstance(n,ast.Call))
    return dict(launches=len(names), kernels=names, calls=dict(calls),
                has_dot=bool(calls['tl.dot']), has_reduction=bool(calls['tl.sum'] or calls['tl.max']))


def audit(root,output):
    rows=[]
    completed=[]
    for path in sorted(Path(root).glob('*/summary.json')):
        summary=read_json(path); manifest=summary['manifest']; directory=path.parent
        if manifest['protocol'].get('search_space')!='source-v1': continue
        completed.append(dict(run=directory.name,status=summary['status'],speedup=summary.get('speedup'),
                              final_candidate=(summary.get('final') or {}).get('candidate_id'),
                              cost=summary.get('estimated_inference_usd',0)))
        for h in summary['history']:
            attempt=h['attempt']; proposed=directory/f'proposal_{attempt:03d}.json'
            proposal=read_json(proposed) if proposed.exists() else {}
            source=proposal.get('source')
            feat={}
            if isinstance(source,str):
                try: feat=features(source,proposal.get('config',{}))
                except (SyntaxError,TypeError,AttributeError): pass
            baseline=(directory/'baseline.py').read_text(encoding='utf-8')
            r=h['result']
            rows.append(dict(run=directory.name,case=manifest['case']['id'],variant=manifest['variant'],seed=manifest['seed'],
                             attempt=attempt,status=r['status'],accepted=h['accepted'],
                             source_supplied=isinstance(source,str),source_differs=bool(source and source!=baseline),
                             structural_screen=bool(h.get('source_edit',{})),
                             error=r.get('error',''),**feat))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'source_audit.json',dict(runs=completed,candidates=rows,
        counts=dict(searches=len(completed),attempts=len(rows),source_supplied=sum(r['source_supplied'] for r in rows),
                    structural_screen=sum(r['structural_screen'] for r in rows),correct=sum(r['status']=='ok' for r in rows),
                    retained=sum(r['accepted'] for r in rows)),
        caveat='Static primitive counts and launch changes require manual review; not causal attribution.'))
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with (output/'source_candidates.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    print(json.dumps(read_json(output/'source_audit.json')['counts']))


def replay_gate(root,output):
    from kernel_study.source_space import GATE_VERSION, check_edit
    rows=[]
    for path in sorted(Path(root).glob('*/events.jsonl')):
        events=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        contexts={e['context']['attempt']:e['context']['incumbent'] for e in events
                  if e['event']=='model_request' and e.get('stage')=='implement'}
        for attempt,parent in sorted(contexts.items()):
            proposal_path=path.parent/f'proposal_{attempt:03d}.json'
            if not proposal_path.exists(): continue
            proposal=read_json(proposal_path)
            row=dict(run=path.parent.name,attempt=attempt)
            try:
                if not isinstance(proposal.get('source'),str):
                    raise ValueError('source_edit_required: complete source is mandatory')
                check_edit(proposal['source'],proposal['config'],parent['source'],parent['config'])
                row['replayed_gate']='pass'
            except (ValueError,SyntaxError,KeyError,TypeError) as exc:
                row.update(replayed_gate='rejected',reason=str(exc))
            rows.append(row)
    write_json(Path(output)/f'gate_v{GATE_VERSION}_replay.json',dict(gate_version=GATE_VERSION,
        scope='Offline AST replay only. Original measurements and search outcomes are unchanged.',
        gate_source_hash=digest((ROOT/'kernel_study/source_space.py').read_text(encoding='utf-8')),candidates=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--replay-gate',action='store_true')
    a=p.parse_args();audit(a.input,a.output)
    if a.replay_gate: replay_gate(a.input,a.output)
