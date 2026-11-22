"""Calibrate and verify the two-round improvement rule."""
import time

from . import bench


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


def first_call(forward):
    import torch
    torch.cuda.synchronize()
    begin = time.perf_counter()
    value = forward()
    torch.cuda.synchronize()
    return value, time.perf_counter() - begin
