"""Measured baseline/transform controls, separate from the LLM experiment."""
from pathlib import Path
from .core import ROOT, baseline_config, cases, environment, event, read_json, utcnow, write_json
from .search import run_worker


def pilot(output, case_id='all', quick=False):
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    protocol = read_json(ROOT/'configs/protocol.json')
    if quick:
        protocol.update(warmup=3,timing_repeats=9,inner_repeats=3,validation_seeds=[101])
    write_json(output/'manifest.json',{'type':'engineering-pilot','environment':environment(),'protocol':protocol,'started_utc':utcnow()})
    rows = []
    for case in cases():
        if case_id != 'all' and case['id'] != case_id:
            continue
        baseline = baseline_config(case['family'])
        for label in ['baseline','transformed']:
            config = dict(baseline)
            if label == 'transformed':
                config['fused'] = True
                config['tensor_core'] = True
                config['block_m'] = config['block_n'] = 32
            result = run_worker({'case':case,'config':config,'incumbent_config':baseline,'protocol':protocol,
                                 'library':True,'profile':True,'final':True},output,case['id']+'_'+label)
            row = {'case':case['id'],'family':case['family'],'label':label,'config':config,'result':result}
            rows.append(row)
            event(output/'events.jsonl','pilot_measurement',**row)
            write_json(output/'pilot_results.json',rows)
            print(case['id'],label,result['status'],result.get('paired_speedup'),flush=True)
    return rows
