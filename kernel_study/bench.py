"""Trusted input generation, references, validation and paired GPU timing."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

from .core import environment, validate_config


def make_inputs(case, seed, edge="random"):
    import torch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    dtype = getattr(torch, case["dtype"])
    shape = case["shape"]
    def rand(shape):
        return torch.randn(shape, generator=gen, device="cuda", dtype=dtype)
    if case["family"] == "matmul":
        m, n, k = shape
        tensors = [rand((m, k)), rand((k, n))]
    else:
        tensors = [rand(shape)]
        if case["family"] == "layernorm":
            tensors += [rand((shape[-1],)), rand((shape[-1],))]
        if case["family"] == "elementwise":
            tensors += [rand((shape[-1],))]
    if edge == "zeros":
        tensors[0].zero_()
    elif edge == "constant":
        tensors[0].fill_(3.)
    elif edge == "large":
        tensors[0].mul_(20.)
    elif edge == "negative":
        tensors[0].copy_(-tensors[0].abs() - 1.)
    return tensors


def reference(case, inputs):
    import torch
    family = case["family"]
    if family == "matmul":
        # FP32 accumulation and output conversion; TF32 disabled by worker.
        return (inputs[0].float() @ inputs[1].float()).to(inputs[0].dtype)
    if family == "softmax":
        return torch.softmax(inputs[0], dim=-1)
    if family == "layernorm":
        return torch.nn.functional.layer_norm(inputs[0], (case["shape"][-1],), inputs[1], inputs[2], 1.e-5)
    return torch.relu(inputs[0] + inputs[1])


def library_reference(case, inputs):
    if case["family"] == "matmul":
        return inputs[0] @ inputs[1]
    return reference(case, inputs)


def prepare(case, config, inputs, source=None):
    if config.get('mode') == 'source-v1':
        from .source_space import prepare as prepare_source
        return prepare_source(case, config, inputs, source)
    import torch
    import triton
    from . import kernels
    from .source_policy import load_source
    validate_config(config)
    mod = load_source(source) if source else kernels
    f = case["family"]
    shape = case["shape"]
    out_shape = shape[:2] if f == "matmul" else shape
    output = torch.empty(out_shape, dtype=inputs[0].dtype, device="cuda")
    scratch = torch.empty((shape[0], 2), device="cuda", dtype=torch.float32)
    intermediate = torch.empty_like(output)
    opts = {"num_warps": config["num_warps"], "num_stages": config["num_stages"]}
    compiled = []
    def record(kernel):
        if all(kernel is not existing for existing in compiled):
            compiled.append(kernel)
    def launch():
        if f == "matmul":
            m, n, k = shape
            bm, bn, bk = (config[x] for x in ("block_m", "block_n", "block_k"))
            kernel = mod.matmul_tensor if config["tensor_core"] else mod.matmul_simt
            record(kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](*inputs, output, m, n, k, bm, bn, bk, **opts))
        else:
            rows, n = shape
            total, block = rows * n, config["block"]
            grid = (triton.cdiv(total, block),)
            rb = triton.next_power_of_2(n)
            if f == "softmax":
                if config["fused"]:
                    record(mod.softmax_fused[(rows,)](inputs[0], output, n, rb, **opts))
                else:
                    record(mod.softmax_stats[(rows,)](inputs[0], scratch, n, rb, **opts))
                    record(mod.softmax_apply[grid](inputs[0], scratch, output, total, n, block, **opts))
            elif f == "layernorm":
                if config["fused"]:
                    record(mod.layernorm_fused[(rows,)](*inputs, output, n, rb, **opts))
                else:
                    record(mod.layernorm_stats[(rows,)](inputs[0], scratch, n, rb, **opts))
                    record(mod.layernorm_apply[grid](*inputs, scratch, output, total, n, block, **opts))
            else:
                if config["fused"]:
                    record(mod.bias_relu[grid](*inputs, output, total, n, block, **opts))
                else:
                    record(mod.bias_add[grid](*inputs, intermediate, total, n, block, **opts))
                    record(mod.relu[grid](intermediate, output, total, block, **opts))
        return output
    return launch, compiled


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


def validate(case, config, seeds, source=None):
    import torch
    records = []
    compile_seconds = 0.
    for index, (seed, edge) in enumerate([(s, "random") for s in seeds] + [(seeds[0], e) for e in ["zeros", "constant", "large", "negative"]]):
        inputs = make_inputs(case, seed, edge)
        snapshots = [x.clone() for x in inputs]
        expected = reference(case, snapshots)
        fn, _ = prepare(case, config, inputs, source)
        begin = time.perf_counter()
        actual = fn()
        torch.cuda.synchronize()
        if index == 0:
            compile_seconds = time.perf_counter() - begin
        # Every candidate must overwrite all output elements on every launch.
        actual.fill_(float('nan'))
        actual = fn()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=case["atol"], rtol=case["rtol"], equal_nan=False)
        if not all(torch.equal(x, y) for x, y in zip(inputs, snapshots)):
            raise AssertionError("candidate mutated an input")
        records.append({"seed": seed, "edge": edge, "max_abs_error": float((actual.float()-expected.float()).abs().max())})
    return records, compile_seconds


def resource_summary(compiled):
    result = []
    for kernel in compiled:
        meta = getattr(kernel, "metadata", None)
        result.append({"registers_per_thread": getattr(kernel, "n_regs", None),
                       "spills": getattr(kernel, "n_spills", None),
                       "shared_bytes": getattr(meta, "shared", None),
                       "num_warps": getattr(meta, "num_warps", None)})
    return result


def evaluate(request):
    import torch
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    case, config, protocol = request["case"], request["config"], request["protocol"]
    result = {"status": "error", "environment": environment()}
    seeds = protocol["final_validation_seeds"] if request.get("final") else protocol["validation_seeds"]
    begin = time.perf_counter()
    result["validation"], result["first_launch_seconds"] = validate(case, config, seeds, request.get("source"))
    result["validation_seconds"] = time.perf_counter() - begin
    inputs = make_inputs(case, seeds[0] + 77)
    fn, compiled = prepare(case, config, inputs, request.get("source"))
    incumbent, _ = prepare(case, request.get("incumbent_config", config), inputs, request.get("incumbent_source", request.get("source")))
    fn(); incumbent(); torch.cuda.synchronize()
    result["candidate"], result["incumbent"] = time_pair(fn, incumbent, protocol)
    speedup = result["incumbent"]["median_ms"] / result["candidate"]["median_ms"]
    result["paired_speedup"] = speedup
    result["confirmed"] = False
    if speedup > protocol["improvement_threshold"]:
        c2, i2 = time_pair(fn, incumbent, protocol)
        result["confirmation"] = {"candidate": c2, "incumbent": i2}
        result["confirmed"] = i2["median_ms"] / c2["median_ms"] > protocol["improvement_threshold"]
    if request.get("library"):
        _, result["library"] = time_pair(fn, lambda: library_reference(case, inputs), protocol)
    if request.get("profile"):
        start = time.perf_counter()
        result["profile"] = {"mode": "compiler-resources", "kernels": resource_summary(compiled),
                             "hardware_counters": None, "note": "No hardware-counter bottleneck claims."}
        result["profile_seconds"] = time.perf_counter() - start
    result["status"] = "ok"
    from .metrics import throughput
    result["throughput"] = throughput(case, result["candidate"]["median_ms"])
    result["worker_seconds"] = time.perf_counter() - begin
    return result
