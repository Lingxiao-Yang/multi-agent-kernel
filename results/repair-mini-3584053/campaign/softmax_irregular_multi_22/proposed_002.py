import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    exp_x = tl.exp(x - mx)
    exp_sum = tl.sum(exp_x, 0)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, exp_sum)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    base_idx = tl.program_id(0) * BLOCK
    offs = tl.arange(0, BLOCK)
    i = base_idx + offs
    mask = i < TOTAL

    row = i // N
    col = i % N

    x = tl.load(X + i, mask, other=0.0)
    mx = tl.load(S + row * 2, mask, other=0.0)
    denom = tl.load(S + row * 2 + 1, mask, other=1.0)

    # Explicit stepwise computation
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    val = exp_x / denom

    tl.store(Y + i, val, mask)
