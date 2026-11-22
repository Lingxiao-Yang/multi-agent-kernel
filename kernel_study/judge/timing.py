"""Paired timing, compiler resources and noise calibration."""
import time


def summarize(values):
    import numpy as np
    return {"median_ms": float(np.median(values)), "p25_ms": float(np.percentile(values, 25)),
            "p75_ms": float(np.percentile(values, 75)), "min_ms": min(values), "samples_ms": values}


def time_pair(a, b, protocol):
    import torch
    for _ in range(protocol["warmup"]):
        a(); b()
    torch.cuda.synchronize()
    samples = [[], []]
    inner = protocol["inner_repeats"]
    # Fixed alternating AB/BA order limits temporal bias; no overlapping work.
    for rep in range(protocol["timing_repeats"]):
        for index in ([0, 1] if rep % 2 == 0 else [1, 0]):
            fn = (a, b)[index]
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(inner):
                fn()
            end.record()
            end.synchronize()
            samples[index].append(start.elapsed_time(end) / inner)
    return [summarize(x) for x in samples]


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
