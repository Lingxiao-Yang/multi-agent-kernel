"""Validate and measure a candidate in its own process."""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import bench
from .core import ROOT, environment, read_json, write_json
from .correctness import check_forward, fp64_chunks, stress_input
from .interface import prepare
from .public_cases import experiment, load_native, task_inputs


def bind(case, source, config, interface, buffers, *, validating=False):
    if interface == 'native':
        native = load_native(source)
        return lambda: native(*buffers), []
    return prepare(case,config,buffers,source,fresh_output=True,poison_output=validating)


def validate(case, source, config, interface, seeds, stress):
    records = []
    samples = [(seed, 'random') for seed in seeds]
    if stress:
        edges = ['zeros','constant','large','negative']
        if case['family']=='layernorm':
            edges.append('large_offset')
        samples.extend((seeds[0], edge) for edge in edges)
    for seed, edge in samples:
        model,inputs,buffers = task_inputs(case,seed)
        if edge!='random':
            stress_input(inputs[0], edge)
        expected = model(*inputs)
        forward,_ = bind(case,source,config,interface,buffers,validating=True)
        first = time.perf_counter()
        reference = fp64_chunks(model,inputs) if edge!='random' else None
        try:
            record = check_forward(lambda *args: forward(),buffers,expected,case,reference_chunks=reference)
        except AssertionError as exc:
            raise AssertionError(f'{edge} seed {seed}: {exc}') from exc
        if not records: first_check_seconds = time.perf_counter()-first
        records.append(dict(seed=seed,edge=edge,**record))
        del model,inputs,buffers,expected,forward
    return records, first_check_seconds


def evaluate(request):
    import torch
    case, protocol = request['case'], request['protocol']
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    source = request['source']
    config = request.get('config')
    interface = request.get('interface','declarative')
    seeds = protocol['final_validation_seeds' if request.get('final') else 'validation_seeds']
    with torch.no_grad():
        records, first_check_seconds = validate(case, source, config, interface, seeds, request.get('final',False))
        validation_seconds = time.perf_counter()-started
        model,inputs,buffers = task_inputs(case,seeds[0]+77)
        forward,compiled = bind(case,source,config,interface,buffers)
        parent,_ = bind(case,request.get('incumbent_source',source),request.get('incumbent_config',config),
                        request.get('incumbent_interface',interface),buffers)
        candidate,incumbent = bench.time_pair(forward,parent,protocol)
        speedup = incumbent['median_ms']/candidate['median_ms']
        result = dict(status='ok',environment=environment(),validation=records,validation_seconds=validation_seconds,
                      first_validation_seconds=first_check_seconds,candidate=candidate,incumbent=incumbent,
                      paired_speedup=speedup,confirmed=False)
        if speedup > protocol['improvement_threshold']:
            c,i = bench.time_pair(forward,parent,protocol)
            result['confirmation'] = {'candidate':c,'incumbent':i}
            result['confirmed'] = i['median_ms']/c['median_ms'] > protocol['improvement_threshold']
        if request.get('library'):
            result['library_candidate'],result['library'] = bench.time_pair(forward,lambda: model(*inputs),protocol)
        if request.get('final') and request.get('compare_compile',True):
            compiled_model = torch.compile(model,mode=experiment()['compile_mode'],fullgraph=True)
            result['compile_candidate'],result['compile'] = bench.time_pair(forward,lambda: compiled_model(*inputs),protocol)
        if request.get('profile'):
            result['profile'] = {'mode':'compiler-resources','kernels':bench.resource_summary(compiled)}
    result['worker_seconds'] = time.perf_counter()-started
    return result


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("output")
    args = p.parse_args()
    start = time.perf_counter()
    try:
        result = evaluate(read_json(args.request))
    except Exception as exc:
        result = {"status": "incorrect" if isinstance(exc, AssertionError) else "error",
                  "error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(), "worker_seconds": time.perf_counter() - start}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
