import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr):
    row = tl.program_id(0)
    BLOCK = 256
    col_block = tl.program_id(1)
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load input slice with masking, padding with -inf for max reduction
    x = tl.load(X + row * N + offs, mask, other=-float('inf'))

    # Step 1: local max and denom per block slice
    local_max = tl.max(x, 0)
    exp_x = tl.exp(x - local_max)
    local_denom = tl.sum(exp_x, 0)

    # Warp-wide reduction across blocks (program_id(1)) for max and denom
    pid1 = tl.program_id(1)
    # Reduce max with warp shuffle down
    for offset in [2, 1]:
        tmp = tl.shfl_down(local_max, offset, 4)
        if pid1 + offset < 4:
            local_max = tl.maximum(local_max, tmp)
    # Reduce denom with warp shuffle down
    for offset in [2, 1]:
        tmp = tl.shfl_down(local_denom, offset, 4)
        if pid1 + offset < 4:
            local_denom += tmp

    # Broadcast final max and denom to all lanes in warp
    max_val = tl.shfl(local_max, 0, 4)
    denom_val = tl.shfl(local_denom, 0, 4)

    # Compute final softmax slice
    y = tl.exp(x - max_val) / denom_val

    # Store result masked
    tl.store(Y + row * N + offs, y, mask)
