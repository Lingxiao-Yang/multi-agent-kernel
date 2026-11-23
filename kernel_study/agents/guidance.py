"""Common language/API correctness guidance, identical across all four arms.

This supplies invariants, not an optimized kernel or hardware tuning recipe.
"""
VERSION = 'native-triton-2026-11-22-v1'
SOURCES = [
    'https://triton-lang.org/main/python-api/generated/triton.language.arange.html',
    'https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html',
    'https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html',
]


CONTRACT = """For implement return JSON with source (the complete Python module) and rationale.
Expose kernel_function with the same tensor arguments as the starting module. Compute in Triton JIT kernels.
The wrapper may check metadata, allocate fresh output and temporary tensors, launch kernels and return the output.
Leave inputs unchanged. Keep each invocation independent. Use torch, triton and triton.language as tl.
Kernel code, launch parameters and constants may all change. Other stages return concise JSON."""


def guidance(case):
    return {
        'version':VERSION, 'sources':SOURCES, 'output_shape':case['output_shape'],
        'rules':[
            'For tl.arange(0, BLOCK), BLOCK is a compile-time power of two. Annotate that kernel argument tl.constexpr. A runtime tensor is not a valid extent.',
            'Logical row width N and padded vector extent RB are different. For a full-row vector use an extent covering every logical column, then mask padding. A smaller extent requires an explicit loop or another complete reduction decomposition.',
            'Mask loads AND stores for logical bounds. Padding must contribute the reduction identity: negative infinity for maximum, zero for sum. After centering padded values for a variance reduction, mask them again so padding does not contribute.',
            'Means and variances divide by logical N, not padded extent. Preserve layernorm epsilon 1e-5. FP32 accumulation and accurately rounded division (tl.div_rn) can matter for constant-input cancellation under the fixed tolerances.',
            'A grid counts program instances, not GPU SMs. Verify the index mapping covers every output. Derive grid and strides from tensor metadata. Allocate temporary storage for all simultaneous partial results.',
            'tl.zeros/tl.full shapes are compile-time tuples, e.g. (BLOCK,), not a scalar integer. Use Triton operations and keep numerical work in device code.',
            'An unchanged program is the starting point. A failed module may be repaired on the next charged attempt.',
        ],
    }


def history_context(history):
    """Preserve every attempt and diagnostic, omit redundant generated code.

The current incumbent and most recent proposal are supplied separately in full.
All original responses remain in events.jsonl; this is only a context view.
"""
    def clean(value):
        if isinstance(value,dict):
            return {k:clean(v) for k,v in value.items() if k not in
                    ['source','config','traceback','samples_ms','validation','environment']}
        if isinstance(value,list): return [clean(x) for x in value]
        if isinstance(value,str) and len(value)>2000: return value[:2000]+' [context truncated; full text logged]'
        return value
    return [clean(row) for row in history]
