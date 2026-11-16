import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    idx = row * N + offs

    # Load input with masking; pad with -inf for max reduction
    x = tl.load(X + idx, offs < N, other=-float('inf'))

    # Step 1: compute max with iterative tree reduction over RB using fixed strides
    m = x
    for stride in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        # Mask out-of-bound elements with -inf
        m2 = tl.where(offs + stride < RB, m[stride:], tl.full((RB - stride,), -float('inf'), dtype=tl.float32))
        m = tl.maximum(m[:stride], m2)
        # Pad m back to length RB with -inf
        m = tl.concat([m, tl.full((RB - stride,), -float('inf'), dtype=tl.float32)])[:RB]

    mx = m[0]

    # Reload input for sum with exp centered by max
    x = tl.load(X + idx, offs < N, other=-float('inf'))
    exp_x = tl.exp(x - mx)
    exp_x = tl.where(offs < N, exp_x, tl.zeros((RB,), dtype=tl.float32))

    # Step 2: compute sum with iterative tree reduction over exp_x
    s = exp_x
    for stride in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        s2 = tl.where(offs + stride < RB, s[stride:], tl.zeros((RB - stride,), dtype=tl.float32))
        s = s[:stride] + s2
        s = tl.concat([s, tl.zeros((RB - stride,), dtype=tl.float32)])[:RB]

    denom = s[0]

    # Step 3: write output softmax values
    y = tl.where(offs < N, exp_x / denom, tl.zeros((RB,), dtype=tl.float32))
    tl.store(Y + idx, y, offs < N)
