"""Validate and measure a candidate in its own process."""
import argparse
import logging
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

from ..records import ROOT, environment, read_json, write_json
from ..tasks.kernelbench import experiment, load_module, task_inputs, task_path
from ..tasks.starts import load_native
from . import timing
from .correctness import check_forward, fp64_chunks, stress_input
from .policy import StaticRejection, check_source


def bind(source, buffers):
    native = load_native(source)
    return lambda: native(*buffers)


def describe_ncu(metrics):
    from kernel_perf_agent.kernel_opt.roofline.ncu_roofline import RooflineAnalyzer
    (last_kernel,) = metrics.values()
    return dict(metrics=metrics, roofline=RooflineAnalyzer().analyze(last_kernel).to_dict(), kernel_selection='last')


def ncu_evidence(source, problem, directory):
    from triton_kernel_agent.opt_worker_component.profiling.kernel_profiler import KernelProfiler
    directory.mkdir(parents=True, exist_ok=True)
    profiler = KernelProfiler(logging.getLogger(__name__), directory, directory)
    sampled = profiler.profile_kernel(source, problem, round_num=0, max_retries=1)
    if sampled is None or not sampled.metrics:
        raise RuntimeError(f'no NCU measurements in {directory}')
    return describe_ncu(sampled.metrics)


def compile_resources(module, buffers):
    from triton.runtime.jit import JITFunction
    compiled = []
    def capture(run):
        @wraps(run)
        def tracked(*args, **kwargs):
            kernel = run(*args, **kwargs)
            if kernel is not None and all(kernel is not item for item in compiled):
                compiled.append(kernel)
            return kernel
        return tracked
    with ExitStack() as patches:
        for jit in vars(module).values():
            if isinstance(jit, JITFunction):
                patches.enter_context(patch.object(jit, 'run', capture(jit.run)))
        module.kernel_function(*buffers)
    return timing.resource_summary(compiled)


def validate(case, source, seeds, stress):
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
        forward = bind(source,buffers)
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
    seeds = protocol['final_validation_seeds' if request.get('final') else 'validation_seeds']
    with torch.no_grad():
        records, first_check_seconds = validate(case, source, seeds, request.get('final',False))
        validation_seconds = time.perf_counter()-started
        model,inputs,buffers = task_inputs(case,seeds[0]+77)
        forward = bind(source,buffers)
        parent = bind(request.get('incumbent_source',source),buffers)
        candidate,incumbent = timing.time_pair(forward,parent,protocol)
        speedup = incumbent['median_ms']/candidate['median_ms']
        result = dict(status='ok',environment=environment(),validation=records,validation_seconds=validation_seconds,
                      first_validation_seconds=first_check_seconds,candidate=candidate,incumbent=incumbent,
                      paired_speedup=speedup,confirmed=False)
        if speedup > protocol['improvement_threshold']:
            c,i = timing.time_pair(forward,parent,protocol)
            result['confirmation'] = {'candidate':c,'incumbent':i}
            result['confirmed'] = i['median_ms']/c['median_ms'] > protocol['improvement_threshold']
        if request.get('library'):
            result['library_candidate'],result['library'] = timing.time_pair(forward,lambda: model(*inputs),protocol)
        if request.get('final') and request.get('compare_compile',True):
            compiled_model = torch.compile(model,mode=experiment()['compile_mode'],fullgraph=True)
            compiled_forward = lambda: compiled_model(*inputs)
            if 'warmup_ms' in protocol:
                value, result['compile_first_seconds'] = timing.first_call(compiled_forward)
                del value
            result['compile_candidate'],result['compile'] = timing.time_pair(forward,compiled_forward,protocol)
        if request.get('profile'):
            result['profile'] = {'mode':'compiler-resources','kernels':compile_resources(load_module(source),buffers)}
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
        proc = subprocess.run([sys.executable, '-m', 'kernel_study.judge.evaluate', str(req), str(out)],
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
        request = read_json(args.request)
        for path in {request['source'],request.get('incumbent_source', request['source'])}:
            check_source(Path(path).read_text())
        result = evaluate(request)
        if request.get('profile') and request['protocol'].get('feedback_mode') == 'timing-and-ncu':
            import torch
            torch.cuda.empty_cache()
            directory = Path(args.output).with_suffix('.ncu')
            if 'shared_ncu_metrics' in request:
                result['profile']['ncu'] = describe_ncu(request['shared_ncu_metrics'])
            else:
                result['profile']['ncu'] = ncu_evidence(Path(request['source']), task_path(request['case']), directory)
            result['profile']['mode'] = 'compiler-resources-and-ncu'
    except StaticRejection as exc:
        result = dict(status='static_rejected',error=str(exc),worker_seconds=time.perf_counter()-start)
    except Exception as exc:
        status = 'incorrect' if isinstance(exc, AssertionError) else 'error'
        if type(exc).__module__.startswith(('triton.compiler','triton.runtime.errors')):
            status = 'compile_error'
        result = {"status": status,
                  "error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(), "worker_seconds": time.perf_counter() - start}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
