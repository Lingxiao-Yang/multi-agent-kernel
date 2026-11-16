import triton
import triton.language as tl

@triton.jit
def softmax_multiwarp(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    # Load with masking
    x = tl.load(X + offs, col < N, other=-float('inf'))
    # Multi-warp reduction for stable max
    mx = tl.max(x, 0)
    # Redundant: triton's reduction is already multi-warp when num_warps>1
    # Exponentiate
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    # Multi-warp reduction for sum
    denom = tl.sum(exp_x, 0)
    # Write normalized output
    y = exp_x / denom
    tl.store(Y + offs, y, col < N)
