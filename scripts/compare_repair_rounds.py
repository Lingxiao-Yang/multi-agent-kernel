"""Paired descriptive analysis of matched mini/full repair pilots.

No two-case bootstrap intervals or causal claims about increased budgets.
Raw data remains immutable; outputs are a separate derived artifact.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kernel_study.core import read_json,write_json,VARIANTS
from kernel_study.analysis import validate_cohort,geometric_mean


def load(root):
    result={}
    for path in sorted(Path(root).glob('*/summary.json')):
        s=read_json(path);m=s['manifest'];key=(m['case']['id'],m['variant'],m['seed'])
        if key in result: raise ValueError('duplicate search')
        if s['status']!='complete' or s['stop_reason']!='candidate_cap':
            raise ValueError('incomplete or non-candidate-limited search')
        if s['budget']['attempts']!=8 or s['budget']['calls']!=32:
            raise ValueError('unmatched budget')
        events=[json.loads(line) for line in (path.parent/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        responses=[e for e in events if e['event']=='model_response']
        if len(responses)!=s['budget']['calls']: raise ValueError('response/call count mismatch')
        usage={k:sum(e['metadata']['usage'][k] for e in responses) for k in ['prompt_tokens','completion_tokens']}
        if usage['prompt_tokens']!=s['budget']['input_tokens'] or usage['completion_tokens']!=s['budget']['output_tokens']:
            raise ValueError('usage mismatch')
        result[key]=(s,path.parent)
    validate_cohort([s for s,_ in result.values()])
    return result


def compare(mini_root,strong_root,output):
    cohorts={'mini':load(mini_root),'strong':load(strong_root)}
    expected={(case,v,seed) for case in ['softmax_irregular','layernorm_irregular'] for v in VARIANTS for seed in [11,22]}
    if set(cohorts['mini'])!=expected or set(cohorts['strong'])!=expected:
        raise ValueError('Expected 16 paired searches per model')
    differences=[]
    for key in cohorts['mini']:
        a,b=[cohorts[label][key][0]['manifest'] for label in ['mini','strong']]
        for field in ['baseline_hash','case_hash','knowledge_hash','code_hashes']:
            if a[field]!=b[field]: raise ValueError('drift: '+field)
        for field in ['gpu','torch','triton','cuda','capability']:
            if a['environment'].get(field)!=b['environment'].get(field): raise ValueError('hardware/software drift: '+field)
        protocol_diff={k for k in set(a['protocol'])|set(b['protocol']) if a['protocol'].get(k)!=b['protocol'].get(k)}
        if protocol_diff-{'model','input_usd_per_million','output_usd_per_million'}:
            raise ValueError('unexpected protocol drift: '+str(protocol_diff))
        differences.append(sorted(protocol_diff))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    rows=[];aggregates={};failure_examples=[]
    for label,cohort in cohorts.items():
        statuses=Counter();prefix={str(k):dict(any_correct=0,any_retained=0) for k in [1,2,4,8]}
        for (case,variant,seed),(s,path) in cohort.items():
            history=s['history'];correct=[h['attempt'] for h in history if h['result']['status']=='ok'];kept=[h['attempt'] for h in history if h['accepted']]
            for k in [1,2,4,8]:
                prefix[str(k)]['any_correct']+=any(a<=k for a in correct)
                prefix[str(k)]['any_retained']+=any(a<=k for a in kept)
            transitions=[b['attempt'] for a,b in zip(history,history[1:]) if a['result']['status']!='ok' and b['result']['status']=='ok']
            statuses.update(h['result']['status'] for h in history)
            final=s['final'];r=final['result'];validation=r['validation']
            expected_validation={(seed,'random') for seed in range(10001,10006)}|{(10001,edge) for edge in ['zeros','constant','large','negative']}
            if r['status']!='ok' or len(validation)!=9 or {(v['seed'],v['edge']) for v in validation}!=expected_validation:
                raise ValueError('unexpected final validation outcome or coverage')
            row=dict(model=label,case=case,variant=variant,seed=seed,speedup=s['speedup'],
                final_candidate=final['candidate_id'],first_correct=min(correct,default=None),
                first_retained=min(kept,default=None),correct_attempts=len(correct),retained_attempts=len(kept),
                recovery_transitions=len(transitions),first_correct_after_2=bool(correct and min(correct)>2),
                first_retained_after_2=bool(kept and min(kept)>2),
                selected_after_2=final['candidate_id']>2,
                input_tokens=s['budget']['input_tokens'],output_tokens=s['budget']['output_tokens'],
                cost_usd=s['estimated_inference_usd'],candidate_ms=r['candidate']['median_ms'],
                library_ms=r['library']['median_ms'],library_over_candidate=r['library']['median_ms']/r['candidate']['median_ms'],
                wall_seconds=s['wall_seconds'])
            prefix_budget=history[1]['budget_after_review'];protocol=s['manifest']['protocol']
            row['cost_through_attempt_2_usd']=(prefix_budget['input_tokens']*protocol['input_usd_per_million']+prefix_budget['output_tokens']*protocol['output_usd_per_million'])/1e6
            row['cost_after_attempt_2_usd']=row['cost_usd']-row['cost_through_attempt_2_usd']
            rows.append(row)
            for h in history:
                if h['result']['status']!='ok':
                    failure_examples.append(dict(model=label,case=case,variant=variant,seed=seed,attempt=h['attempt'],
                        status=h['result']['status'],error=h['result'].get('error','')[:2500]))
        subset=[r for r in rows if r['model']==label]
        aggregates[label]=dict(searches=len(subset),attempts_by_status=dict(statuses),prefix_searches=prefix,
            final_improved=sum(r['final_candidate']>0 for r in subset),geomean_speedup=geometric_mean([r['speedup'] for r in subset]),
            first_correct_after_2=sum(r['first_correct_after_2'] for r in subset),
            first_retained_after_2=sum(r['first_retained_after_2'] for r in subset),
            selected_after_2=sum(r['selected_after_2'] for r in subset),
            searches_with_recovery=sum(r['recovery_transitions']>0 for r in subset),
            input_tokens=sum(r['input_tokens'] for r in subset),output_tokens=sum(r['output_tokens'] for r in subset),
            calls=sum(s['budget']['calls'] for s,_ in cohort.values()),cost_usd=sum(r['cost_usd'] for r in subset),
            cost_after_attempt_2_usd=sum(r['cost_after_attempt_2_usd'] for r in subset),
            by_case={case:dict(improved=sum(r['final_candidate']>0 for r in subset if r['case']==case),
                geomean_speedup=geometric_mean([r['speedup'] for r in subset if r['case']==case])) for case in sorted({r['case'] for r in subset})},
            by_variant={v:dict(improved=sum(r['final_candidate']>0 for r in subset if r['variant']==v),
                geomean_speedup=geometric_mean([r['speedup'] for r in subset if r['variant']==v])) for v in VARIANTS})
    paired=[]
    for key in sorted(cohorts['mini']):
        a,b=[cohorts[label][key][0] for label in ['mini','strong']]
        paired.append(dict(case=key[0],variant=key[1],seed=key[2],speedup_ratio=b['speedup']/a['speedup']))
    report=dict(matched=True,allowed_protocol_differences=differences[0],models=aggregates,
        paired_model_geomean_ratio=geometric_mean([x['speedup_ratio'] for x in paired]),
        pairing=paired,caveat='Two cases and two seeds: descriptive results only; model phases ran sequentially on separate Slurm allocations. Prefix retention is not independently fresh-validated at each prefix. No causal effect of guidance or extra attempts is isolated.')
    write_json(out/'comparison.json',report);write_json(out/'failure_examples.json',failure_examples)
    with (out/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plot(rows,aggregates,out/'repair_comparison.png')
    print(json.dumps(report,indent=2)[:12000])
    return report


def plot(rows,aggregates,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    colors={'mini':'#4679b5','strong':'#d16c32'}
    for label in ['mini','strong']:
        a=aggregates[label];counts=[a['prefix_searches'][str(k)]['any_retained'] for k in [1,2,4,8]]
        axes[0].plot([1,2,4,8],counts,marker='o',label=label,color=colors[label])
    axes[0].set(xlabel='Candidate-attempt prefix',ylabel='Searches with a retained candidate',xticks=[1,2,4,8],yticks=[0,4,8,12,16],ylim=(-.3,16.5))
    for i,case in enumerate(['softmax_irregular','layernorm_irregular']):
        ax=axes[i+1]
        for offset,label in [(-.14,'mini'),(.14,'strong')]:
            for j,v in enumerate(VARIANTS):
                values=[r['speedup'] for r in rows if r['model']==label and r['case']==case and r['variant']==v]
                ax.scatter([j+offset]*len(values),values,label=label if j==0 else None,color=colors[label],alpha=.85)
        ax.axhline(1,color='gray',linestyle='--',linewidth=1)
        ax.set(title=case.replace('_irregular',''),ylabel='Final validated speedup',xticks=range(4),xticklabels=['single','single+K','multi','multi+K'])
    axes[0].legend();axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('A100 guided repair pilots: two seeds; descriptive evidence')
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mini',required=True);p.add_argument('--strong',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();compare(args.mini,args.strong,args.output)
