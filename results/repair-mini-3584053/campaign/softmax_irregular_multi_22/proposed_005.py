import triton
import triton.language as tl

@triton.jit
def softmax_exp_sum(X, S, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, RB)
    mask = col < N

    x = tl.load(X + row * N + col, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    tl.store(S + row * 4, mx)

    exp_x = tl.exp(x - mx)
    exp_sum = tl.sum(exp_x, 0)
    tl.store(S + row * 4 + 1, exp_sum)

@triton.jit
def softmax_sum_reduce(S, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    mx = tl.load(S + row * 4)
    exp_sum = tl.load(S + row * 4 + 1)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, exp_sum)

@triton.jit
def softmax_apply(X, S, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, RB)
    mask = col < N

    x = tl.load(X + row * N + col, mask, other=0.0)
    mx = tl.load(S + row * 2)  # scalar load without mask
    denom = tl.load(S + row * 2 + 1)  # scalar load without mask

    val = tl.exp(x - mx) / denom
    tl.store(Y + row * N + col, val, mask)
