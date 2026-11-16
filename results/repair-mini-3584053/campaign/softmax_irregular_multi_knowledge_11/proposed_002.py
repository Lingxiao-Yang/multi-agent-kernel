import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N

    # Load input with masking, pad with -inf for max reduction
    x = tl.load(X + row * N + offs, mask, other=-float('inf'))

    # Warp-level max reduction over the tile (lane-wise max)
    mx = tl.max(x, 0)  # lane local max
    # Hierarchical cross-lane warp max reductions
    mx = tl.max_contiguous(mx)
    mx = tl.max_contiguous(mx)

    # Broadcast max to all lanes
    exps = tl.where(mask, tl.exp(x - mx), 0.0)

    # Warp-level sum reduction over exps (lane-wise sum)
    s = tl.sum(exps, 0)  # lane local sum
    # Hierarchical cross-lane warp sum reductions
    s = tl.sum(s)
    s = tl.sum(s)

    # Normalize
    y = tl.where(mask, exps / s, 0.0)

    # Store result with mask
    tl.store(Y + row * N + offs, y, mask)
