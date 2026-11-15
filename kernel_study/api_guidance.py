"""Common language/API correctness guidance, identical across all four arms.

This supplies invariants, not an optimized kernel or hardware tuning recipe.
"""
VERSION = 'triton-correctness-2026-11-15-v1'
SOURCES = [
    'https://triton-lang.org/main/python-api/generated/triton.language.arange.html',
    'https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html',
    'https://triton-lang.org/main/python-api/generated/triton.language.div_rn.html',
]


def guidance(case):
    rows,n = case['shape'][:2]
    return {
        'version':VERSION, 'sources':SOURCES,
        'resolved_scalars':dict(ROWS=rows,M=rows,N=n,TOTAL=rows*n,RB=1 << (n-1).bit_length(),
                               **({'K':case['shape'][2]} if case['family']=='matmul' else {})),
        'rules':[
            'For tl.arange(0, BLOCK), BLOCK is a compile-time power of two. Annotate that kernel argument tl.constexpr. A runtime tensor is not a valid extent.',
            'Logical row width N and padded vector extent RB are different. For a full-row vector use an extent covering every logical column, then mask padding. A smaller extent requires an explicit loop or another complete reduction decomposition.',
            'Mask loads AND stores for logical bounds. Padding must contribute the reduction identity: negative infinity for maximum, zero for sum. After centering padded values for a variance reduction, mask them again so padding does not contribute.',
            'Means and variances divide by logical N, not padded extent. Preserve layernorm epsilon 1e-5. FP32 accumulation and accurately rounded division (tl.div_rn) can matter for constant-input cancellation under the fixed tolerances.',
            'A grid counts program instances, not GPU SMs. Verify the index mapping covers every output. Use the provided scalar symbols in args to avoid manual shape-arithmetic errors; grid entries remain concrete integers.',
            'tl.zeros/tl.full shapes are compile-time tuples, e.g. (BLOCK,), not a scalar integer. Only primitives listed in launch_contract are allowed; do not invent explicit shared-memory APIs or synchronization calls.',
            'Code/config-only cosmetic rewrites do not count as optimization. You must author changed device computation or data movement; a failed authored kernel may be repaired on the next charged attempt.',
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
