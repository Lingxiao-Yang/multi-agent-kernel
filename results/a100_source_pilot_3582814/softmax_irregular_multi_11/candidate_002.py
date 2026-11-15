import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    mask = col < N
    x = tl.load(X + offs, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    denom = tl.sum(tl.exp(x - mx), 0)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOTAL
    row = idx // N
    col = idx % N
    x = tl.load(X + idx, mask, other=0.0)
    mx = tl.load(S + row * 2, mask, other=-float('inf'))
    denom = tl.load(S + row * 2 + 1, mask, other=1.0)
    safe_denom = tl.where(denom > 0, denom, 1.0)
    y = tl.exp(x - mx) / safe_denom
    y = tl.where(mask, y, 0.0)
    tl.store(Y + idx, y, mask)
