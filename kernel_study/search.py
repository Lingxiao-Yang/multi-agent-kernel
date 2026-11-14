from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .core import (ROOT, Budget, baseline_config, digest, environment, event, launch_contract, read_json,
                   utcnow, validate_config, write_json)
from .knowledge import select_cards
from .source_policy import check_source

GENERAL = ('You are a general GPU kernel optimizer. Use the supplied evidence to optimize the current incumbent. '
           'Do not invent measurements. All decisions about retention are made by the harness. Return JSON only.')
ROLES = {
    'analyze': 'You are the performance analyst. Rank bottleneck hypotheses using measured evidence and state uncertainty. Return JSON only.',
    'advise': 'You are the hardware advisor. Propose transformations and resource trade-offs from the shared evidence. Return JSON only.',
    'implement': 'You are the kernel implementer. Apply the shared plan to the incumbent. Return JSON only.',
    'review': 'You are the result reviewer. Explain measured outcomes and failures and suggest next directions. You cannot decide retention. Return JSON only.'
}
CONTRACT = '''For stage implement, return {"config": <complete config>, "source": null or complete Triton module string, "rationale": "..."}.
Config keys/domains: num_warps [1,2,4,8], num_stages [1,2,3,4], block [64,128,256,512,1024], block_m/block_n [16,32,64], block_k [16,32,64], fused bool, tensor_core bool.
source=null preserves the incumbent source. Source edits may only change @triton.jit device functions; preserve names and argument signatures. Only imports triton and triton.language as tl are allowed. No host functions, file/network access, autotune, reference/timer modification or input mutation. Each config tested costs one candidate. Other stages return a concise JSON object.'''


def run_worker(request, directory, label):
    directory = Path(directory)
    req = directory / (label + '.request.json')
    out = directory / (label + '.result.json')
    write_json(req, request)
    env = {k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ['API_KEY','SECRET','TOKEN','PASSWORD'])}
    begin = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, '-m', 'kernel_study.worker', str(req), str(out)],
                   cwd=ROOT, env=env, capture_output=True, text=True, timeout=request['protocol']['worker_timeout_s'])
        (directory / (label + '.stderr.log')).write_text(proc.stderr, encoding='utf-8')
        result = read_json(out) if out.exists() else {'status':'worker_crash','returncode':proc.returncode, 'error':proc.stderr[-4000:]}
    except subprocess.TimeoutExpired:
        result = {'status':'timeout', 'error':'worker exceeded wall-time limit'}
    result['process_seconds'] = time.perf_counter() - begin
    write_json(out, result)
    return result


def compact_result(result):
    def clean(value):
        if isinstance(value, dict):
            return {k:clean(v) for k,v in value.items() if k not in ['samples_ms','traceback','validation']}
        if isinstance(value, list):
            return [clean(x) for x in value]
        return value
    return clean(result)


def search(case, variant, seed, protocol, provider, output, drop_role=None, drop_category=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    log = output / 'events.jsonl'
    start = time.perf_counter()
    starting_cost = getattr(provider, 'spent', 0.)
    starting_completed_cost = getattr(provider, 'completed_cost', 0.)
    source_text = (ROOT / 'kernel_study/kernels.py').read_text(encoding='utf-8')
    # Candidate source stores only the device module; trusted wrapper is never editable.
    base_source = output / 'baseline.py'
    base_source.write_text(source_text, encoding='utf-8')
    budget = Budget(**{k:protocol[k] for k in ['candidate_cap','call_cap','token_safety_cap']})
    manifest = {'started_utc':utcnow(), 'case':case, 'variant':variant, 'seed':seed,
                'provider':provider.name, 'model':provider.model, 'protocol':protocol,
                'protocol_hash':digest(protocol), 'case_hash':digest(case), 'baseline_hash':digest(source_text),
                'knowledge_hash':digest(read_json(ROOT/'knowledge/cards.json')), 'environment':environment(),
                'drop_role':drop_role, 'drop_category':drop_category,
                'scheduler_job_id':os.environ.get('SLURM_JOB_ID'), 'queue_wait_seconds':None,
                'code_hashes':{str(p.relative_to(ROOT)):digest(p.read_text(encoding='utf-8')) for p in sorted((ROOT/'kernel_study').glob('*.py'))}}
    write_json(output/'manifest.json', manifest)
    event(log, 'run_start', manifest=manifest)
    base_config = baseline_config(case['family'])
    request = {'case':case,'config':base_config,'source':str(base_source),'protocol':protocol,'library':True,'profile':True}
    base_result = run_worker(request, output, 'baseline')
    event(log, 'baseline', result=base_result)
    if base_result['status'] != 'ok':
        write_json(output/'summary.json', {'status':'baseline_failed','manifest':manifest,'budget':budget.snapshot()})
        return read_json(output/'summary.json')
    incumbent = {'id':0, 'config':base_config, 'source':str(base_source), 'result':base_result}
    retained = [incumbent]
    history = []
    profile_count = 1
    stop = None
    resolved_model = None
    def call(stage, context):
        nonlocal resolved_model
        budget.reserve_call()
        system = ROLES[stage] if variant.startswith('multi') else GENERAL
        if stage == drop_role:
            # Same call slot reassigned to implementer/general reasoning.
            replacement = 'advise' if stage == 'implement' else 'implement'
            system = ROLES[replacement] if variant.startswith('multi') else GENERAL
        prompt = system + '\n' + CONTRACT
        event(log, 'model_request', stage=stage, system=prompt, context=context, budget=budget.snapshot())
        reply = provider.complete(prompt, context, stage, seed + budget.calls)
        budget.charge_tokens(reply.input_tokens, reply.output_tokens)
        event(log, 'model_response', stage=stage, text=reply.text, model=reply.model, metadata=reply.metadata, budget=budget.snapshot())
        if resolved_model is not None and reply.model != resolved_model:
            raise RuntimeError('model_version_changed')
        resolved_model = reply.model
        return json.loads(reply.text)
    while not budget.stop_reason():
        budget.reserve_attempt()
        attempt = budget.attempts
        cards = select_cards(case['family'], incumbent['result'].get('profile'), drop_category) if 'knowledge' in variant else []
        context = {'case':case,'attempt':attempt,'incumbent':{'id':incumbent['id'],'config':incumbent['config'],
                   'source':Path(incumbent['source']).read_text(encoding='utf-8'), 'result':compact_result(incumbent['result'])},
                   'history':history, 'knowledge_cards':cards, 'hardware':manifest['environment'],
                   'launch_contract':launch_contract(case['family']),
                   'api_documentation':['https://triton-lang.org/main/python-api/triton.language.html'], 'budget':budget.snapshot()}
        candidate = None
        try:
            for stage in ['analyze','advise']:
                context[stage] = call(stage, context)
            candidate = call('implement', context)
            validate_config(candidate['config'])
            text = candidate.get('source') or context['incumbent']['source']
            check_source(text)
            candidate_source = output / f'candidate_{attempt:03d}.py'
            candidate_source.write_text(text, encoding='utf-8')
            candidate['source'] = str(candidate_source)
            candidate['id'], candidate['parent_id'] = attempt, incumbent['id']
            result = run_worker({'case':case,'config':candidate['config'],'source':candidate['source'],
                        'incumbent_config':incumbent['config'],'incumbent_source':incumbent['source'],
                        'protocol':protocol}, output, f'candidate_{attempt:03d}')
            accepted = result['status'] == 'ok' and result['confirmed']
            if accepted:
                candidate['result'] = result
                incumbent = candidate
                retained.append(candidate)
                if profile_count < protocol['profile_cap']:
                    prof = run_worker({'case':case,'config':candidate['config'],'source':candidate['source'],
                                      'protocol':protocol,'profile':True}, output, f'profile_{attempt:03d}')
                    profile_count += 1
                    result['profile'] = prof.get('profile', {'unavailable':prof['status']})
            record = {'attempt':attempt,'parent_id':candidate['parent_id'],'config':candidate['config'],
                      'source_hash':digest(text),'rationale':candidate.get('rationale'), 'accepted':accepted,
                      'result':compact_result(result),'cards':[c['id'] for c in cards], 'budget':budget.snapshot()}
        except Exception as exc:
            record = {'attempt':attempt,'accepted':False,'result':{'status':'proposal_error','error':str(exc)}, 'budget':budget.snapshot()}
            # Network/usage/budget problems are not silently retried or imputed.
            if isinstance(exc, (OSError, RuntimeError)):
                stop = str(exc)
        history.append(record)
        event(log, 'candidate', **record)
        if stop is None and budget.calls < budget.call_cap and budget.tokens < budget.token_safety_cap:
            try:
                context['measured_result'] = record
                review = call('review', context)
                record['review'] = review
            except Exception as exc:
                event(log, 'review_error', error=str(exc), budget=budget.snapshot())
                stop = str(exc)
        record['budget_after_review'] = budget.snapshot()
        write_json(output/'state.json', {'incumbent':incumbent,'history':history,'budget':budget.snapshot()})
        if stop:
            break
    # Every retained candidate is eligible for fallback; baseline remains included.
    final = None
    for item in reversed(retained):
        checked = run_worker({'case':case,'config':item['config'],'source':item['source'],
                    'incumbent_config':base_config,'incumbent_source':str(base_source),'protocol':protocol,
                    'final':True,'library':True}, output, f'final_{item["id"]:03d}')
        event(log, 'final_validation', candidate_id=item['id'], result=checked)
        if checked['status'] == 'ok':
            final = {'candidate_id':item['id'], 'result':checked, 'config':item['config']}
            break
    final_stop = stop or budget.stop_reason()
    completed_search = final_stop in ['candidate_cap','call_cap','token_safety_cap']
    summary = {'status':('complete' if completed_search else 'interrupted') if final else 'final_failed', 'manifest':manifest,
               'stop_reason':stop or budget.stop_reason(), 'budget':budget.snapshot(), 'profile_invocations':profile_count,
               'final':final,'history':history,'wall_seconds':time.perf_counter()-start,'resolved_model':resolved_model,
               'scientific_evidence':provider.name == 'llm' and completed_search and resolved_model is not None,
               'estimated_inference_usd':getattr(provider,'completed_cost',0)-starting_completed_cost,
               'budget_committed_usd':getattr(provider,'spent',0)-starting_cost}
    if final:
        summary['speedup'] = final['result']['paired_speedup']
        # Initial incumbent retained => measured fluctuations don't imply a selected regression.
        if final['candidate_id'] == 0:
            summary['speedup'] = 1.
    write_json(output/'summary.json', summary)
    event(log, 'run_end', status=summary['status'], budget=budget.snapshot())
    return summary
