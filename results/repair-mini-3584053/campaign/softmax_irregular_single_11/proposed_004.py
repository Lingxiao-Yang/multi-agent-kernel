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
        valid = offs + stride < RB
        m2 = tl.where(valid, m[stride:], tl.full((RB - stride,), -float('inf'), dtype=tl.float32))
        m = tl.maximum(m[:stride], m2)
        # Pad m back to length RB with -inf using tuple syntax for tl.concat
        m = tl.concat((m, tl.full((RB - stride,), -float('inf'), dtype=tl.float32)))[:RB]

    mx = m[0]

    # Reload input for sum with exp centered by max
    x = tl.load(X + idx, offs < N, other=-float('inf'))
    exp_x = tl.exp(x - mx)
    exp_x = tl.where(offs < N, exp_x, tl.zeros((RB,), dtype=tl.float32))

    # Step 2: compute sum with iterative tree reduction over exp_x
    s = exp_x
    for stride in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
        valid = offs + stride < RB
        s2 = tl.where(valid, s[stride:], tl.zeros((RB - stride,), dtype=tl.float32))
        s = s[:stride] + s2
        # Pad s back to length RB with zeros using tuple syntax for tl.concat
        s = tl.concat((s, tl.zeros((RB - stride,), dtype=tl.float32)))[:RB]

    denom = s[0]

    # Step 3: write output softmax values
    y = tl.where(offs < N, exp_x / denom, tl.zeros((RB,), dtype=tl.float32))
    tl.store(Y + idx, y, offs < N)
