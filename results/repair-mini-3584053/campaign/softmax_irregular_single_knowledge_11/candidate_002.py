import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs
    ptr = row * N + col
    mask = col < N

    # Load with masking; out-of-bounds elements get -inf
    x = tl.load(X + ptr, mask, other=-float('inf'))

    # Two-stage max reduction for large BLOCK
    half = BLOCK // 2
    x1 = x[:half]
    x2 = x[half:BLOCK]
    mx1 = tl.max(x1, 0)
    mx2 = tl.max(x2, 0)
    mx = tl.maximum(mx1, mx2)

    # Compute exp(x - max) with masking
    exps = tl.exp(x - mx) * mask

    # Two-stage sum reduction
    sum1 = tl.sum(exps[:half], 0)
    sum2 = tl.sum(exps[half:BLOCK], 0)
    denom = sum1 + sum2

    # Normalize
    y = exps / denom

    # Store output with masking
    tl.store(Y + ptr, y, mask)
