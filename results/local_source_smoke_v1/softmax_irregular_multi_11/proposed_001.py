import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptrs = row * N + offs
    mask = offs < N
    x = tl.load(X + ptrs, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    exps = tl.exp(x - mx)
    denom = tl.sum(exps, 0)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    i = pid * BLOCK + offs
    mask = i < TOTAL
    row = i // N
    x = tl.load(X + i, mask, other=0)
    mx = tl.load(S + row * 2, mask, other=0)
    denom = tl.load(S + row * 2 + 1, mask, other=1)
    y = tl.exp(x - mx) / denom
    tl.store(Y + i, y, mask)
