import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    idx = row * N + offs

    # Load input with masking, pad with -inf for max reduction
    x = tl.load(X + idx, offs < N, other=-float('inf'))

    # Iterative tree max reduction over padded RB vector without concat
    m = x
    stride = RB // 2
    while stride > 0:
        valid = offs + stride < RB
        m2 = tl.where(valid, m[stride:], tl.full((RB - stride,), -float('inf'), dtype=tl.float32))
        m = tl.maximum(m[:stride], m2)
        m = tl.concat((m[:stride], tl.full((RB - stride,), -float('inf'), dtype=tl.float32)))[:RB]
        stride //= 2

    mx = m[0]

    # Reload input for sum reduction, center by max
    x = tl.load(X + idx, offs < N, other=-float('inf'))
    exp_x = tl.exp(x - mx)
    exp_x = tl.where(offs < N, exp_x, tl.zeros((RB,), dtype=tl.float32))

    # Iterative tree sum reduction over exp_x without concat
    s = exp_x
    stride = RB // 2
    while stride > 0:
        valid = offs + stride < RB
        s2 = tl.where(valid, s[stride:], tl.zeros((RB - stride,), dtype=tl.float32))
        s = s[:stride] + s2
        s = tl.concat((s[:stride], tl.zeros((RB - stride,), dtype=tl.float32)))[:RB]
        stride //= 2

    denom = s[0]

    # Store max and sum per row
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    mask = i < TOTAL

    x = tl.load(X + i, mask, other=0)
    mx = tl.load(S + row * 2, mask, other=0)
    denom = tl.load(S + row * 2 + 1, mask, other=1)

    y = tl.exp(x - mx) / denom
    tl.store(Y + i, y, mask)
