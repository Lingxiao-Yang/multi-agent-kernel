"""Describe budget prefixes and failure-to-correct transitions without imputing wins."""
from collections import Counter
from pathlib import Path
from .core import VARIANTS, read_json, write_json


def summarize(root, output, plan):
    paths=sorted(Path(root).glob('*/summary.json'))
    expected={(case,v,seed) for case in plan['cases'] for v in VARIANTS for seed in plan['seeds']}
    seen=set(); rows=[]; failures=Counter()
    for path in paths:
        s=read_json(path);m=s['manifest'];key=(m['case']['id'],m['variant'],m['seed'])
        if key in seen: raise ValueError('duplicate case/variant/seed')
        seen.add(key)
        hist=s.get('history',[])
        correct=[r['attempt'] for r in hist if r['result']['status']=='ok']
        accepted=[r['attempt'] for r in hist if r['accepted']]
        recoveries=[b['attempt'] for a,b in zip(hist,hist[1:])
                    if a['result']['status']!='ok' and b['result']['status']=='ok']
        failures.update(r['result']['status'] for r in hist)
        rows.append(dict(case=key[0],variant=key[1],seed=key[2],status=s['status'],
                         first_correct=min(correct,default=None),first_retained=min(accepted,default=None),
                         recovery_transition_attempts=recoveries,
                         final_candidate=(s.get('final') or {}).get('candidate_id'),speedup=s.get('speedup'),
                         prefixes={str(k):dict(correct=sum(a<=k for a in correct),retained=sum(a<=k for a in accepted))
                                   for k in [1,2,4,8]},
                         attempts=s['budget']['attempts'],calls=s['budget']['calls'],tokens=s['budget']['tokens']))
    complete=seen==expected and all(r['status']=='complete' and r['attempts']==plan['attempts'] for r in rows)
    per_case={case:sum(r['case']==case and r['status']=='complete' and (r['final_candidate'] or 0)>0 for r in rows)
              for case in plan['cases']}
    report=dict(complete=complete,searches=len(rows),expected_searches=len(expected),
                improved_searches_by_case=per_case,
                stronger_required=(any(n<plan['minimum_improved_searches_per_case'] for n in per_case.values()) if complete else None),
                attempts_by_status=dict(failures),runs=rows,
                first_correct_after_attempt_2=sum(r['first_correct'] is not None and r['first_correct']>2 for r in rows),
                searches_with_recovery=sum(bool(r['recovery_transition_attempts']) for r in rows),
                caveat='Recovery transitions are temporal observations, not proof that the next proposal directly repaired the prior source. Prefix correctness is search validation, not independent fresh validation at every prefix.')
    write_json(output,report)
    return report
