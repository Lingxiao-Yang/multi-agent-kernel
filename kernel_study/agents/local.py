from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ..budget import Budget, interruption
from ..judge.evaluate import run_worker
from ..judge.select import screen_candidate, select_final
from ..records import ROOT, code_hashes, digest, environment, event, read_json, utcnow, write_json
from ..tasks import kernelbench
from ..tasks.starts import start
from .guidance import CONTRACT, VERSION, guidance, history_context
from .knowledge import select_cards

GENERAL = ('You are a general GPU kernel optimizer. Use the supplied evidence to optimize the current incumbent. '
           'Do not invent measurements. All decisions about retention are made by the harness. Return JSON only.')
ROLES = {
    'analyze': 'You are the performance analyst. Rank bottleneck hypotheses using measured evidence and state uncertainty. Return JSON only.',
    'advise': 'You are the hardware advisor. Propose transformations and resource trade-offs from the shared evidence. Return JSON only.',
    'implement': 'You are the kernel implementer. Apply the shared plan to the incumbent. Return JSON only.',
    'review': 'You are the result reviewer. Explain measured outcomes and failures and suggest next directions. You cannot decide retention. Return JSON only.'
}

STAGE_CONTRACTS = {
    'analyze': ('Return a JSON object with an analysis field. Describe the bottleneck hypotheses and supporting '
                'measurements in plain language. Include no source code or pseudocode.'),
    'advise': ('Return a JSON object with an advice field containing at most three short sentences in plain language. '
               'Describe the proposed edits and their trade-offs. Include no source code or pseudocode.'),
    'implement': (CONTRACT + '\nSubmit a module different from the current incumbent. If it already looks optimal, '
                  'try the most promising untested direction supported by the evidence.'),
    'review': ('Return a JSON object with diagnosis, outcome and next_step fields. Refer to the measurements '
               'available for this attempt. Include no source code or pseudocode.'),
}


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
    started = time.perf_counter()
    starting_cost = getattr(provider, 'spent', 0.)
    starting_completed_cost = getattr(provider, 'completed_cost', 0.)
    shared_guidance = None
    if protocol.get('shared_api_guidance'):
        if protocol['shared_api_guidance'] != VERSION:
            raise ValueError('unknown shared API guidance version')
        shared_guidance = guidance(case)
    source_text = start(case)
    case = dict(case)
    if 'problem' not in case:
        case['problem'] = kernelbench.task_path(case).read_text()
    if 'test' not in case:
        case['test'] = kernelbench.test_source(case)
    (output/'problem.py').write_text(case['problem'])
    (output/'test.py').write_text(case['test'])
    base_source = output / 'baseline.py'
    base_source.write_text(source_text, encoding='utf-8')
    budget = Budget(**{k:protocol[k] for k in ['candidate_cap','call_cap','token_safety_cap']})
    manifest = {'started_utc':utcnow(), 'case':case, 'variant':variant, 'seed':seed,
                'provider':provider.name, 'model':provider.model, 'protocol':protocol,
                'protocol_hash':digest(protocol), 'case_hash':digest(case), 'baseline_hash':digest(source_text),
                'knowledge_hash':digest(read_json(ROOT/'knowledge/cards.json')), 'environment':environment(),
                'drop_role':drop_role, 'drop_category':drop_category,
                'scheduler_job_id':os.environ.get('SLURM_JOB_ID'), 'queue_wait_seconds':None,
                'code_hashes':code_hashes()}
    write_json(output/'manifest.json', manifest)
    if shared_guidance:
        write_json(output/'shared_api_guidance.json', shared_guidance)
    event(log, 'run_start', manifest=manifest)
    request = {'case':case,'interface':'native','source':str(base_source),'protocol':protocol,'library':True,'profile':True}
    if 'shared_ncu_metrics' in protocol:
        request['shared_ncu_metrics'] = protocol['shared_ncu_metrics']
    base_result = run_worker(request, output, 'baseline')
    event(log, 'baseline', result=base_result)
    if base_result['status'] != 'ok':
        write_json(output/'summary.json', {'status':'baseline_failed','manifest':manifest,'budget':budget.snapshot()})
        return read_json(output/'summary.json')
    incumbent = {'id':0, 'source':str(base_source), 'result':base_result}
    baseline = incumbent
    history = []
    last_proposal = None
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
        prompt = system + '\n' + STAGE_CONTRACTS[stage]
        event(log, 'model_request', stage=stage, system=prompt, context=context, budget=budget.snapshot())
        reply = provider.complete(prompt, context, stage, seed + budget.calls)
        budget.charge_tokens(reply.input_tokens, reply.output_tokens)
        event(log, 'model_response', stage=stage, text=reply.text, model=reply.model, metadata=reply.metadata, budget=budget.snapshot())
        if resolved_model is not None and reply.model != resolved_model:
            raise RuntimeError('model_version_changed')
        resolved_model = reply.model
        answer = json.loads(reply.text)
        if stage != 'implement' and isinstance(answer, dict):
            answer.pop('source', None)
        return answer
    while not budget.stop_reason():
        budget.reserve_attempt()
        attempt = budget.attempts
        cards = select_cards(case['family'], incumbent['result'].get('profile'), drop_category) if 'knowledge' in variant else []
        context = {'case':case,'attempt':attempt,'incumbent':{'id':incumbent['id'],
                   'source':Path(incumbent['source']).read_text(encoding='utf-8'), 'result':compact_result(incumbent['result'])},
                   'history':history_context(history) if shared_guidance else history,
                   'knowledge_cards':cards, 'hardware':dict(manifest['environment'],specs=kernelbench.experiment()['gpu_specs']),
                   'api_documentation':['https://triton-lang.org/main/python-api/triton.language.html'], 'budget':budget.snapshot()}
        if incumbent['id'] != 0:
            context['starting_module'] = source_text
        candidate = None
        if shared_guidance:
            context['shared_api_guidance'] = shared_guidance
            context['stage_instructions'] = {
                'analyze':'Inspect the current code and previous failure. Distinguish compiler/API mistakes, output coverage, and numerical errors from performance hypotheses.',
                'advise':'Give a concise implementable plan. If the previous proposal failed, identify the exact repair and verify shape/constexpr/masking invariants before proposing more optimization.',
                'implement':'Return the complete native module and a rationale. Check output coverage, strides and reduction precision. Repairs consume this attempt; no unmeasured speedup claims.',
                'review':'Return concise diagnosis, outcome, and next_step fields. Explain the measured result. Only the implementation stage supplies an executable module.'}
        if last_proposal is not None:
            context['previous_proposal'] = last_proposal
        try:
            for stage in ['analyze','advise']:
                context[stage] = call(stage, context)
            candidate = call('implement', context)
            last_proposal = dict(candidate)
            write_json(output / f'proposal_{attempt:03d}.json', candidate)
            text = candidate.get('source')
            if not isinstance(text, str) or not text.strip():
                raise ValueError('source_edit_required: submit the complete native module')
            candidate_source = output / f'candidate_{attempt:03d}.py'
            candidate_source.write_text(text, encoding='utf-8')
            if not isinstance(candidate.get('rationale'), str) or not candidate['rationale'].strip():
                raise ValueError('provide a rationale for the change')
            candidate['source'] = str(candidate_source)
            candidate['id'], candidate['parent_id'] = attempt, incumbent['id']
            result = screen_candidate(text,source_text,incumbent_source=context['incumbent']['source'])
            if result is None:
                result = run_worker({'case':case,'interface':'native','source':candidate['source'],
                                     'incumbent_source':incumbent['source'],'protocol':protocol}, output, f'candidate_{attempt:03d}')
            accepted = result['status'] == 'ok' and result['confirmed']
            if accepted:
                candidate['result'] = result
                incumbent = candidate
                if profile_count < protocol['profile_cap']:
                    prof = run_worker({'case':case,'interface':'native','source':candidate['source'],
                                      'protocol':protocol,'profile':True}, output, f'profile_{attempt:03d}')
                    profile_count += 1
                    result['profile'] = prof.get('profile', {'unavailable':prof['status']})
            record = {'attempt':attempt,'parent_id':candidate['parent_id'],'interface':'native',
                      'source_hash':digest(text),'rationale':candidate.get('rationale'), 'accepted':accepted,
                      'previous_failure_attempt':history[-1]['attempt'] if history and history[-1]['result']['status']!='ok' else None,
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
                context['proposed_candidate'] = last_proposal
                review = call('review', context)
                record['review'] = review
            except Exception as exc:
                event(log, 'review_error', error=str(exc), budget=budget.snapshot())
                stop = str(exc)
        record['budget_after_review'] = budget.snapshot()
        write_json(output/'state.json', {'incumbent':incumbent,'history':history,'budget':budget.snapshot()})
        if stop:
            break
    candidates = [dict(id=h['attempt'], source=str(output/f"candidate_{h['attempt']:03d}.py"))
                  for h in history if h['result']['status']=='ok']
    final, selection = select_final(output, case, protocol, 'native', baseline, candidates)
    final_stop = stop or budget.stop_reason()
    completed_search = final_stop in ['candidate_cap','call_cap','token_safety_cap']
    summary = {'status':('complete' if completed_search else 'interrupted') if final else 'final_failed', 'manifest':manifest,
               'stop_reason':stop or budget.stop_reason(), 'budget':budget.snapshot(), 'profile_invocations':profile_count,
               'final':final,'history':history,'wall_seconds':time.perf_counter()-started,'resolved_model':resolved_model,
               'scientific_evidence':provider.name == 'llm' and completed_search and resolved_model is not None,
               'estimated_inference_usd':getattr(provider,'completed_cost',0)-starting_completed_cost,
               'budget_committed_usd':getattr(provider,'spent',0)-starting_cost}
    reason = interruption(output)
    summary['selection'] = selection
    if hasattr(provider,'ledger'):
        usage = provider.ledger.summary()
        summary['budget'].update(usage)
        summary['estimated_inference_usd'] = usage['completed_usd']
        summary['budget_committed_usd'] = usage['committed_usd']
    if reason:
        summary.update(status='interrupted', stop_reason=reason)
    if final:
        summary['speedup'] = final['result']['paired_speedup']
        # Initial incumbent retained => measured fluctuations don't imply a selected regression.
        if final['candidate_id'] == 0:
            summary['speedup'] = 1.
    write_json(output/'summary.json', summary)
    event(log, 'run_end', status=summary['status'], budget=budget.snapshot())
    return summary
