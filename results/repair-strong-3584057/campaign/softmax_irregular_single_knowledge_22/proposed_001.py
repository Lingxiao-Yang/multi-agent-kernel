import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptrs = X + row * N + offs
    # First, compute the row max with masking
    x = tl.load(x_ptrs, col_mask, other=-float('inf'))
    mx = tl.max(x, 0)
    # Compute exp(x - mx) with masking
    exp_x = tl.exp(x - mx) * col_mask
    # Compute denom with masking
    denom = tl.sum(exp_x, 0)
    # Avoid division by zero
    denom = tl.where(denom == 0, 1, denom)
    y = exp_x / denom
    # Store with masking
    tl.store(Y + row * N + offs, y, col_mask)
