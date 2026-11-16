import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    mask = col < N
    # Load input with masking; pad with -inf for max reduction
    x = tl.load(X + offs, mask, other=-float('inf'))
    # Compute max per row
    mx = tl.max(x, 0)
    # Compute exp(x - max) with zero masked padding
    ex = tl.exp(x - mx) * mask
    # Compute sum per row
    denom = tl.sum(ex, 0)
    # Final output with masking
    y = ex / denom
    tl.store(Y + offs, y, mask)
