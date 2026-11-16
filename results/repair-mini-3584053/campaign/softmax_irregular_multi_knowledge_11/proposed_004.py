import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N

    # Load input with masking, pad with -inf for max reduction
    x = tl.load(X + row * N + offs, mask, other=-float('inf'))

    # Step 1: lane-wise max reduction
    lane_max = tl.max(x, 0)  # lane-local max

    # Step 2: warp-level max reduction over lane_max
    warp_max = tl.max_contiguous(lane_max)

    # Step 3: cross-warp max reduction over warp_max
    max_val = tl.max(warp_max)

    # Broadcast max_val to all lanes
    exps = tl.where(mask, tl.exp(x - max_val), 0.0)

    # Step 1: lane-wise sum reduction
    lane_sum = tl.sum(exps, 0)  # lane-local sum

    # Step 2: warp-level sum reduction over lane_sum
    warp_sum = tl.sum(lane_sum)

    # Step 3: cross-warp sum reduction over warp_sum
    sum_val = tl.sum(warp_sum)

    # Normalize
    y = tl.where(mask, exps / sum_val, 0.0)

    # Store result with mask
    tl.store(Y + row * N + offs, y, mask)
