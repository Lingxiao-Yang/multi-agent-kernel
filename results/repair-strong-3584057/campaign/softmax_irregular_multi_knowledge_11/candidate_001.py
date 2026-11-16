import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    # Load input row with masking
    x = tl.load(X + offs, col < N, other=-float('inf'))
    # Compute maximum value in the row for numerical stability
    mx = tl.max(x, 0)
    x_centered = x - mx
    # Exponentiate and compute denominator
    exp_x = tl.exp(x_centered)
    denom = tl.sum(exp_x, 0)
    # Normalize
    y = exp_x / denom
    # Store softmax output, mask tail
    tl.store(Y + offs, y, col < N)
