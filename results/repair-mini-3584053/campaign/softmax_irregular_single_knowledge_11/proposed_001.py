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

    # Compute max for the row
    mx = tl.max(x, 0)

    # Compute exp(x - max) with masking
    exps = tl.exp(x - mx) * mask

    # Compute sum of exps
    denom = tl.sum(exps, 0)

    # Normalize
    y = exps / denom

    # Store output with masking
    tl.store(Y + ptr, y, mask)
