import triton
import triton.language as tl

@triton.jit
def softmax_multiwarp_hier(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    # Load row input with masking
    x = tl.load(X + offs, col < N, other=-float('inf'))
    # Hierarchical reduction for max: each warp computes local max, then warp 0 reduces
    warp_id = tl.program_id(1) if tl.num_warps() > 1 else 0
    lane = tl.arange(0, BLOCK) % 32
    max_local = tl.max(x, 0)
    # Emulate multi-warp: share local max between warps via registers (serial reduction)
    max_warps = tl.full((tl.num_warps(),), -float('inf'), dtype=tl.float32)
    max_warps = tl.where(warp_id == 0, max_local, max_warps)
    for w in range(1, tl.num_warps()):
        warp_mask = (warp_id == w)
        max_warps = tl.where(warp_mask, max_local, max_warps)
    # Serial reduction of max_warps (since Triton cannot communicate between warps directly)
    max_row = max_warps[0]
    for w in range(1, tl.num_warps()):
        max_row = tl.maximum(max_row, max_warps[w])
    x_centered = x - max_row
    exp_x = tl.exp(x_centered)
    # Hierarchical reduction for sum: each warp computes local sum, then warp 0 reduces
    sum_local = tl.sum(exp_x, 0)
    sum_warps = tl.zeros((tl.num_warps(),), dtype=tl.float32)
    sum_warps = tl.where(warp_id == 0, sum_local, sum_warps)
    for w in range(1, tl.num_warps()):
        warp_mask = (warp_id == w)
        sum_warps = tl.where(warp_mask, sum_local, sum_warps)
    sum_row = sum_warps[0]
    for w in range(1, tl.num_warps()):
        sum_row = sum_row + sum_warps[w]
    y = exp_x / sum_row
    tl.store(Y + offs, y, col < N)
