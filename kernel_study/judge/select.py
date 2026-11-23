"""Select public finalists and measure a pair on the same inputs."""
import ast
import fcntl
import json
from pathlib import Path

from ..records import read_json, write_json
from .evaluate import run_worker
from .policy import StaticRejection, check_source


def changed_program(source, start):
    def statements(text):
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and ast.get_docstring(node) is not None:
                node.body.pop(0)
        return ast.dump(tree)
    return statements(source) != statements(start)


def screen_candidate(source, start):
    if not source.strip():
        return {'status':'no_submission'}
    try:
        check_source(source)
    except StaticRejection as exc:
        return {'status':'static_rejected','error':str(exc)}
    if not changed_program(source, start):
        return {'status':'unchanged'}
    return None


def select_final(output, case, protocol, interface, start, candidates):
    retained = [start]
    checks = {}

    def evaluate(item, parent, label, libraries):
        request = dict(case=case, protocol=protocol, interface=interface, final=True, library=libraries,
                       compare_compile=libraries, source=item['source'],
                       incumbent_source=parent['source'])
        return run_worker(request, output, label)

    baseline = Path(start['source']).read_text()
    for item in candidates:
        attempt = item['id']
        point = dict(attempt=attempt, selected=False)
        if not changed_program(Path(item['source']).read_text(), baseline):
            point['status'] = 'unchanged'
        else:
            result = evaluate(item, retained[-1], f'audit_{attempt:03d}', False)
            point['status'] = result['status']
            if result['status'] == 'ok':
                point['latency_ms'] = result['candidate']['median_ms']
                if result['confirmed']:
                    retained.append(item)
                    point['selected'] = True
        checks[attempt] = point
    for item in reversed(retained):
        result = evaluate(item, start, f'final_{item["id"]:03d}', True)
        if result['status'] == 'ok' and (item['id'] == 0 or result['confirmed']):
            return dict(candidate_id=item['id'], result=result, source=Path(item['source']).name,
                        interface=interface), list(checks.values())
        if item['id']:
            checks[item['id']]['endpoint_status'] = 'unconfirmed' if result['status']=='ok' else result['status']
    return None, list(checks.values())


def audit(output, agent):
    summary = read_json(output/'summary.json')
    manifest, final = summary['manifest'], summary.get('final')
    log = output/'model_events.jsonl'
    events = [json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
    replies = {e['id']:e['budget'] for e in events if e['event']=='response'}
    checked = {p['attempt']:p for p in summary.get('selection', [])}
    history = {h['attempt']:h for h in summary.get('history', [])}
    candidates = []
    for call in events:
        if call['event'] != 'request' or call['stage'] not in ('implement','proposal','repair'):
            continue
        budget = replies.get(call['id'], call['budget'])
        i = call['budget']['attempts']
        status = history.get(i, {}).get('result', {}).get('status','no_submission')
        candidates.append({'attempt':i, 'status':status, 'usd':budget['committed_usd'], **checked.get(i, {})})
    correct_new = any('latency_ms' in c and c.get('endpoint_status') in (None, 'unconfirmed') for c in candidates)
    row = dict(agent=agent, case=manifest['case']['public_id'], seed=manifest['seed'], status=summary['status'],
               stop_reason=summary.get('stop_reason'), correct_new=correct_new,
               candidates=candidates, budget=summary['budget'], cost_usd=summary.get('estimated_inference_usd',0.),
               committed_usd=summary.get('budget_committed_usd',0.), wall_seconds=summary.get('wall_seconds',0.))
    row['evaluator_process_seconds'] = sum(read_json(p).get('process_seconds',0.) for p in output.glob('*.result.json'))
    row['gpu_slot_seconds'] = row['wall_seconds']
    if 'upstream_selected' in summary:
        row['upstream_selected'] = summary['upstream_selected']
    if final:
        r = final['result']
        row.update(latency_ms=r['candidate']['median_ms'], fallback=final['candidate_id']==0,
                   initial_ms=r['incumbent']['median_ms'], speedup=r['paired_speedup'],
                   compile_speedup=r['compile']['median_ms']/r['compile_candidate']['median_ms'],
                   finalist={k:final[k] for k in ('source','interface')})
    write_json(output/'audit.json', row)
    return row


def measure_pair(directory, single, greedy):
    if not all((p/'audit.json').exists() for p in (single,greedy)):
        return None
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'pair.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        audits = [read_json(p/'audit.json') for p in (single,greedy)]
        if any(row['status']!='complete' for row in audits):
            return None
        result_file = directory/'direct.result.json'
        if result_file.exists():
            previous = read_json(result_file)
            if previous['status']=='ok':
                return previous
        local, native = (row['finalist'] for row in audits)
        if any(item['interface']!='native' for item in (local,native)):
            raise ValueError('archived launch-plan pairs require their registered evaluator')
        manifest = read_json(single/'summary.json')['manifest']
        request = dict(case=manifest['case'], protocol=manifest['protocol'], source=str(single/local['source']),
                       interface='native', incumbent_source=str(greedy/native['source']))
        return run_worker(request, directory, 'direct')
