import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offs = row * N + cols
    mask = cols < N
    # Load input row with padding masked
    x = tl.load(X + offs, mask, other=-float('inf'))
    # Compute row maximum for numerical stability
    mx = tl.max(x, 0)
    # Center and exponentiate
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    # Compute denominator (partition function)
    denom = tl.sum(exp_x, 0)
    # Normalize
    y = exp_x / denom
    # Store results to output, mask padding
    tl.store(Y + offs, y, mask)
