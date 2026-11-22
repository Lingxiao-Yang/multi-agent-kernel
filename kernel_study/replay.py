"""Replay archived reductions and measure the two-round improvement rule."""
import argparse
import time
from pathlib import Path

from . import bench
from .core import ROOT, baseline_config, cases, digest, environment, read_json, write_json


def archived_entries(root):
    for path in sorted(Path(root).glob('repair-*/campaign/*/summary.json')):
        run = read_json(path)
        case = run['manifest']['case']['id']
        start = read_json(path.with_name('baseline.request.json'))['config']
        final = run['final']
        index = final['candidate_id']
        history = {item['attempt']: item for item in run['history']}
        for role, i, config in [('start', 0, start), ('final', index, final['config'])]:
            source = path.with_name(f'candidate_{i:03}.py' if i else 'baseline.py')
            sha = digest(source.read_text())
            expected = history[i]['source_hash'] if i else run['manifest']['baseline_hash']
            if sha != expected:
                raise ValueError(f'source hash mismatch: {source}')
            yield dict(case=case, campaign=path.parents[2].name, run=path.parent.name,
                       role=role, candidate_id=i, config=config, source=str(source),
                       source_hash=sha, key=digest([sha,config]), archived_status=final['result']['status'])


def threshold_from_controls(ratios, floor):
    return max(floor, max(min(pair) for pair in ratios))


def confirmed(ratios, threshold):
    return min(ratios) > threshold


def noise_trials(a, b, protocol, count):
    trials = []
    for _ in range(count):
        rounds = [bench.time_pair(a, b, protocol) for _ in range(2)]
        trials.append({'rounds': rounds, 'ratios': [y['median_ms']/x['median_ms'] for x,y in rounds]})
    return trials


def calibrate(a, protocol, count):
    trials = noise_trials(a, a, protocol, count)
    threshold = threshold_from_controls([t['ratios'] for t in trials], protocol['improvement_threshold'])
    return {'threshold': threshold, 'trials': trials}


def check_noise(a, b, protocol, count, threshold):
    trials = noise_trials(a, b, protocol, count)
    return dict(threshold=threshold, trials=trials,
                confirmations=sum(confirmed(t['ratios'],threshold) for t in trials))


def noise(case, protocol, phase, output):
    import torch
    inputs = bench.make_inputs(case, 7701 if phase == 'calibrate' else 8801)
    from .source_space import baseline
    source, config = baseline(case)
    path = output / (case['id'] + '.py')
    path.write_text(source)
    start, _ = bench.prepare(case, config, inputs, path, fresh_output=True)
    if phase == 'calibrate':
        return calibrate(start, protocol, 20)
    frozen = read_json(output / 'calibrate.json')['cases'][case['id']]['threshold']
    control = {**baseline_config(case['family']), 'fused': True}
    fused, _ = bench.prepare(case, control, inputs, fresh_output=True)
    torch.testing.assert_close(fused(), bench.reference(case,inputs), atol=case['atol'], rtol=case['rtol'])
    return {'self': check_noise(start, start, protocol, 20, frozen),
            'fused': check_noise(fused, start, protocol, 5, frozen)}


def errors(actual, expected, case):
    import torch
    if actual.shape != expected.shape:
        raise ValueError('incorrect output shape')
    difference = (actual.double() - expected.double()).abs()
    finite = torch.isfinite(difference)
    bad = ~finite | (difference > case['atol'] + case['rtol'] * expected.double().abs())
    return {'bad': int(bad.sum()), 'nonfinite': int((~finite).sum()),
            'max_abs_error': float(difference[finite].max()) if finite.any() else None}


def stress(case, item):
    import torch
    records = {}
    for label,offset,scale in [('large_offset',1024.,1/32), ('low_variance',64.,1/8),
                               ('near_constant',1.,1/65536)]:
        x, weight, bias = bench.make_inputs(case, 9901)
        x.mul_(scale).add_(offset)
        inputs = [x,weight,bias]
        snapshots = [v.clone() for v in inputs]
        xd = x.double()
        centered = xd - xd.mean(-1, keepdim=True)
        reference64 = centered * torch.rsqrt(centered.square().mean(-1,keepdim=True) + 1e-5)
        reference64 = reference64 * weight.double() + bias.double()
        eager = bench.reference(case, inputs)
        run, _ = bench.prepare(case, item['config'], inputs, item['source'])
        run().fill_(float('nan'))
        actual = run()
        records[label] = {'candidate_vs_eager': errors(actual, eager, case),
                          'candidate_vs_fp64': errors(actual, reference64, case),
                          'eager_vs_fp64': errors(eager, reference64, case),
                          'inputs_unchanged': all(torch.equal(x,y) for x,y in zip(inputs,snapshots))}
    return records


def first_call(forward):
    import torch
    torch.cuda.synchronize()
    begin = time.perf_counter()
    value = forward()
    torch.cuda.synchronize()
    return value, time.perf_counter() - begin


def replay(case, entries, protocol):
    import torch
    from tritonbench.operators.softmax.operator import triton_softmax_fn
    from tritonbench.operators.layer_norm.tutorial import layer_norm
    inputs = bench.make_inputs(case, 7759)
    snapshots = [x.clone() for x in inputs]
    unique = {x['key']: x for x in entries}
    starts = {x['key'] for x in entries if x['role']=='start'}
    if len(starts) != 1 or len(entries) != 32:
        raise ValueError(f"expected a common start and 16 archived searches for {case['id']}")
    functions = {k:bench.prepare(case, x['config'], inputs, x['source'], fresh_output=True)[0]
                 for k,x in unique.items()}
    start = functions[next(iter(starts))]
    eager = lambda: bench.reference(case, inputs)
    functions['eager'] = eager
    functions['compile'] = torch.compile(eager, mode='max-autotune-no-cudagraphs', fullgraph=True)
    if case['family'] == 'softmax':
        functions['tritonbench'] = lambda: triton_softmax_fn(inputs[0])
    else:
        functions['tritonbench'] = lambda: layer_norm(inputs[0], (case['shape'][1],), *inputs[1:], 1e-5)
    expected = eager()
    results = {}
    for k,forward in functions.items():
        value, seconds = first_call(forward)
        timing, baseline = bench.time_pair(forward, start, protocol)
        results[k] = {'first_call_seconds': seconds, 'forward': timing, 'start': baseline,
                      'paired_speedup': baseline['median_ms'] / timing['median_ms'],
                      'correctness': errors(value,expected,case),
                      'inputs_unchanged': all(torch.equal(a,b) for a,b in zip(inputs,snapshots))}
        if case['family'] == 'layernorm' and k in unique:
            results[k]['stress'] = stress(case,unique[k])
    mapping = [{k:v for k,v in x.items() if k not in ('source','config')} for x in entries]
    return {'entries': mapping, 'measurements': results}


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['replay','calibrate','hold'])
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    protocol = read_json(ROOT/'configs/protocol.json')
    report = {'protocol':protocol, 'environment':environment(), 'cases':{}}
    entries = list(archived_entries(args.archive)) if args.phase == 'replay' else []
    with torch.no_grad():
        for case in cases():
            if case['id'] not in ('softmax_irregular','layernorm_irregular'):
                continue
            value = (replay(case,[e for e in entries if e['case']==case['id']],protocol)
                     if args.phase=='replay' else noise(case,protocol,args.phase,args.output))
            report['cases'][case['id']] = value
            write_json(args.output/(args.phase+'.json'),report)
            print(args.phase,case['id'],flush=True)


if __name__ == '__main__':
    main()
