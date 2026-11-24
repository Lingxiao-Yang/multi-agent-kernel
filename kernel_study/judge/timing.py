"""Paired timing, compiler resources and noise calibration."""
from collections import deque
from statistics import median
import time


def summarize(values):
    import numpy as np
    return {"median_ms": float(np.median(values)), "p25_ms": float(np.percentile(values, 25)),
            "p75_ms": float(np.percentile(values, 75)), "min_ms": min(values), "samples_ms": values}


def cuda_milliseconds(forward, repeats, events):
    start, end = events
    start.record()
    for _ in range(repeats):
        forward()
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def time_pair(a, b, protocol):
    import torch
    forwards = (a, b)
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    warm = [deque(maxlen=5), deque(maxlen=5)]
    work_ms = 0.
    pairs = 0
    if 'warmup_ms' in protocol:
        for forward in forwards:
            forward()
            torch.cuda.synchronize()
        while work_ms < protocol['warmup_ms']:
            for side, forward in enumerate(forwards):
                elapsed = cuda_milliseconds(forward, 1, events)
                warm[side].append(elapsed)
                work_ms += elapsed
            pairs += 1
    else:
        for _ in range(protocol['warmup']):
            a()
            b()
        torch.cuda.synchronize()

    samples = [[], []]
    inner = protocol['inner_repeats']
    for rep in range(protocol['timing_repeats']):
        for side in ((0, 1) if rep % 2 == 0 else (1, 0)):
            elapsed = cuda_milliseconds(forwards[side], inner, events)
            samples[side].append(elapsed / inner)
    result = [summarize(side) for side in samples]
    for side, observed in enumerate(warm):
        if observed:
            reference = median(observed)
            limit = protocol['slow_sample_ratio'] * reference
            result[side].update(warmup_gpu_ms=work_ms, warmup_pairs=pairs,
                                warmup_median_ms=reference, slow_sample_threshold_ms=limit,
                                slow_sample_indices=[i for i,ms in enumerate(samples[side]) if ms > limit])
    return result


def resource_summary(compiled):
    result = []
    for kernel in compiled:
        meta = getattr(kernel, "metadata", None)
        result.append({"registers_per_thread": getattr(kernel, "n_regs", None),
                       "spills": getattr(kernel, "n_spills", None),
                       "shared_bytes": getattr(meta, "shared", None),
                       "num_warps": getattr(meta, "num_warps", None)})
    return result


def threshold_from_controls(ratios, floor):
    return max(floor, max(min(pair) for pair in ratios))


def confirmed(ratios, threshold):
    return min(ratios) > threshold


def noise_trials(a, b, protocol, count):
    trials = []
    for _ in range(count):
        rounds = [time_pair(a, b, protocol) for _ in range(2)]
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


def first_call(forward):
    import torch
    torch.cuda.synchronize()
    begin = time.perf_counter()
    value = forward()
    torch.cuda.synchronize()
    return value, time.perf_counter() - begin
