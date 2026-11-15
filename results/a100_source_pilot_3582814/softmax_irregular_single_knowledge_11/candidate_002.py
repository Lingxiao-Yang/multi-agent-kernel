import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    exps = tl.exp(x - mx)
    denom = tl.sum(exps, 0)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    base_idx = tl.program_id(0) * BLOCK
    offsets = tl.arange(0, BLOCK)
    i = base_idx + offsets
    mask = i < TOTAL
    row = i // N
    x = tl.load(X + i, mask, other=0.0)
    mx = tl.load(S + row * 2, mask, other=0.0)
    denom = tl.load(S + row * 2 + 1, mask, other=1.0)
    res = tl.exp(x - mx) / denom
    tl.store(Y + i, res, mask)
