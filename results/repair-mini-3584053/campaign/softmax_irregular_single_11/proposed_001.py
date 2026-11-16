import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    idx = row * N + offs

    # Load input with masking; pad with -inf for max reduction
    x = tl.load(X + idx, offs < N, other=-float('inf'))

    # Step 1: compute max with iterative tree reduction over RB
    m = x
    stride = RB // 2
    while stride > 0:
        # For strides that cross boundary, mask with -inf
        m2 = tl.where(offs + stride < RB, m[stride:], tl.full((RB - stride,), -float('inf'), dtype=tl.float32))
        m = tl.maximum(m[:stride], m2)
        # Keep only first stride elements for next iteration
        m = tl.concatenate([m, tl.full((RB - stride,), -float('inf'), dtype=tl.float32)])[:RB]
        stride //= 2

    mx = m[0]

    # Load again for sum with exp centered by max
    x = tl.load(X + idx, offs < N, other=-float('inf'))
    exp_x = tl.exp(x - mx)
    exp_x = tl.where(offs < N, exp_x, tl.zeros((RB,), dtype=tl.float32))

    # Step 2: compute sum with iterative tree reduction over exp_x
    s = exp_x
    stride = RB // 2
    while stride > 0:
        s2 = tl.where(offs + stride < RB, s[stride:], tl.zeros((RB - stride,), dtype=tl.float32))
        s = s[:stride] + s2
        s = tl.concatenate([s, tl.zeros((RB - stride,), dtype=tl.float32)])[:RB]
        stride //= 2

    denom = s[0]

    # Step 3: write output softmax values
    y = tl.where(offs < N, exp_x / denom, tl.zeros((RB,), dtype=tl.float32))
    tl.store(Y + idx, y, offs < N)
