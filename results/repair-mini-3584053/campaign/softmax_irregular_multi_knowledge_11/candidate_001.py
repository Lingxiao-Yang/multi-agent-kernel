import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N

    # Load input with masking, pad with -inf for max reduction
    x = tl.load(X + row * N + offs, mask, other=-float('inf'))

    # Warp-level max reduction over the tile
    mx = tl.max(x, 0)
    mx = tl.max_contiguous(mx)  # cross-lane warp max

    # Broadcast max to all lanes
    # Subtract max and exponentiate, mask padded elements to zero
    exps = tl.where(mask, tl.exp(x - mx), 0.0)

    # Warp-level sum reduction over exps
    s = tl.sum(exps, 0)
    s = tl.sum(s)  # cross-warp sum

    # Normalize
    y = tl.where(mask, exps / s, 0.0)

    # Store result with mask
    tl.store(Y + row * N + offs, y, mask)
