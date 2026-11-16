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

    # Hierarchical max reduction: reduce within warp lanes in two stages
    # Stage 1: partial max reduction over 512 elements per warp lane (assuming BLOCK=1024)
    max1 = tl.max(x[:BLOCK//2], 0)
    max2 = tl.max(x[BLOCK//2:], 0)
    max_val = tl.maximum(max1, max2)

    # Pipeline stage 2: warp-level max reduction across max_val
    max_val = tl.max(max_val, 0)

    # Compute exponentials with masking
    ex = tl.exp(x - max_val) * mask

    # Hierarchical sum reduction in two stages
    sum1 = tl.sum(ex[:BLOCK//2], 0)
    sum2 = tl.sum(ex[BLOCK//2:], 0)
    denom = sum1 + sum2

    # Pipeline stage 2: warp-level sum reduction across denom
    denom = tl.sum(denom, 0)

    # Compute final output
    y = ex / denom
    tl.store(Y + offs, y, mask)
