import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    idx = row * N + offs
    mask = offs < N
    x = tl.load(X + idx, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    exp_x = tl.exp(x - mx)
    denom = tl.sum(exp_x, 0)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    row = i // N
    x = tl.load(X + i, mask, other=0)
    mx = tl.load(S + row * 2, mask, other=0)
    denom = tl.load(S + row * 2 + 1, mask, other=1)
    y = tl.exp(x - mx) / denom
    tl.store(Y + i, y, mask)
