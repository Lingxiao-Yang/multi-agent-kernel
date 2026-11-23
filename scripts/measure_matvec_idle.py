"""Measure both archived matvec programs after controlled periods of GPU idle time."""
import argparse
from contextlib import contextmanager
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import time


def schedule():
    rng = random.Random(771)
    conditions = list(product(('start', 'greedy'), ('self', 'eager', 'compile'), (0, .5, 2, 5)))
    for repeat in range(5):
        block = conditions.copy()
        rng.shuffle(block)
        for program, reference, idle in block:
            yield repeat, program, reference, idle


@contextmanager
def telemetry(output, uuid):
    fields = 'timestamp,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,pstate'
    with (output/'telemetry.csv').open('x') as csv:
        process = subprocess.Popen(['nvidia-smi', '-i', uuid, '--query-gpu='+fields,
                                    '--format=csv,nounits', '--loop-ms=50'], stdout=csv)
        try:
            yield
            if process.poll() is not None:
                raise RuntimeError('GPU telemetry stopped during sampling')
        finally:
            process.terminate()
            process.wait()


def trial(torch, left, right, idle):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    while True:
        left()
        right()
        end.record()
        end.synchronize()
        if start.elapsed_time(end) >= 750:
            break
    before_idle = time.time_ns()
    if idle:
        time.sleep(idle)
    samples = []
    functions = (left, right)
    for index in range(60):
        for side in ((0, 1) if index % 2 == 0 else (1, 0)):
            host_start_ns = time.time_ns()
            start.record()
            functions[side]()
            end.record()
            end.synchronize()
            samples.append([index, side, host_start_ns, time.time_ns(), start.elapsed_time(end)])
    return before_idle, samples


def main():
    import torch
    from kernel_study.records import digest, read_json, write_json
    from kernel_study.tasks import kernelbench
    from kernel_study.campaign.preflight import check_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plan = kernelbench.experiment()
    device = check_device(plan)
    args.output.mkdir(mode=0o700)
    preflight = args.archive/'preflight/4'
    greedy = args.archive/'campaign/smoke-4-greedy-771'
    paths = {'start':preflight/'input.py', 'greedy':greedy/'proposal_001.py'}
    hashes = {k:digest(p.read_text()) for k,p in paths.items()}
    if hashes != {'start':read_json(preflight/'preflight.json')['native_source_hash'],
                  'greedy':read_json(greedy/'summary.json')['history'][0]['source_hash']}:
        raise ValueError('matvec source differs from archived measurement')
    write_json(args.output/'manifest.json', dict(device=device, sources=hashes, schedule=list(schedule()),
               sample_columns=['round','side','host_start_ns','host_end_ns','cuda_ms'],
               conditioning_ms=750, samples_per_side=60, seed=7713))
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad(), telemetry(args.output, device['uuid']):
        model, inputs, buffers = kernelbench.task_inputs(plan['cases']['4'],7713)
        compiled = torch.compile(model,mode=plan['compile_mode'],fullgraph=True)
        programs = {k:kernelbench.load_module(path).kernel_function for k,path in paths.items()}
        forwards = {k:(lambda fn=fn: fn(*buffers)) for k,fn in programs.items()}
        forwards.update(eager=lambda: model(*inputs),compile=lambda: compiled(*inputs))
        for fn in forwards.values():
            fn()
        torch.cuda.synchronize()
        with (args.output/'samples.jsonl').open('x', buffering=1) as stream:
            for repeat, name, reference, idle in schedule():
                left = forwards[name]
                right = left if reference=='self' else forwards[reference]
                before, samples = trial(torch, left, right, idle)
                print(json.dumps(dict(repeat=repeat,program=name,reference=reference,idle_s=idle,
                                      before_idle_ns=before,samples=samples)),file=stream)


if __name__ == '__main__':
    main()
