"""Paired forward timing and compiler resources."""


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
