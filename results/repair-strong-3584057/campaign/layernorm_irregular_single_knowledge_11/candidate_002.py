import triton
import triton.language as tl

@triton.jit
def layernorm_split_row_vector(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    cols = split * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    offs = row * N + cols
    # Load input partial row
    x = tl.load(X + offs, mask=mask, other=0.0)
    # Compute local partials for sum and sum of squares
    psum = tl.sum(x, axis=0)
    psum_sq = tl.sum(x * x, axis=0)
    # Write partials to output scratch (in output Y's tail region)
    # Each row stores its two partials in Y at Y[row * N + N + split*2]
    # But since we cannot use extra memory, both splits redundantly compute both partials
    # Exchange via registers: both splits compute both partials for their own half
    # Now, each split must fetch the other's partial results: we do this by launching with grid=(rows,2) and using tl.sum to combine both
    # Each split computes the full sum by summing over both splits' psum/psum_sq
    psum0 = tl.sum(x, axis=0) if split == 0 else 0.0
    psum1 = tl.sum(x, axis=0) if split == 1 else 0.0
    psum_all = tl.full((), 0.0, tl.float32)
    psum_sq0 = tl.sum(x * x, axis=0) if split == 0 else 0.0
    psum_sq1 = tl.sum(x * x, axis=0) if split == 1 else 0.0
    # Both splits redundantly read the other half to get the complete sum
    # Instead, just have both splits compute their own partial, and then in both splits, load the other split's half to sum
    # This is not possible without shared memory/atomics, so in practice, both splits recompute the full sum for their row
    # Instead, both splits load the entire row in two steps and sum, but since we are limited to BLOCK=512 per split, we must loop
    # So: both splits independently accumulate both partials for the full row
    # Step 1: accumulate full sum for the row via two steps
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)
    for i in range(2):
        sub_cols = i * BLOCK + tl.arange(0, BLOCK)
        sub_mask = sub_cols < N
        sub_offs = row * N + sub_cols
        sub_x = tl.load(X + sub_offs, mask=sub_mask, other=0.0)
        sum_x += tl.sum(sub_x, axis=0)
        sum_x2 += tl.sum(sub_x * sub_x, axis=0)
    mean = tl.div_rn(sum_x, N)
    var = tl.div_rn(sum_x2, N) - mean * mean
    inv_std = tl.rsqrt(var + 1e-5)
    # Normalize only our region
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    x_centered = x - mean
    y = (x_centered * inv_std) * w + b
    tl.store(Y + offs, y, mask=mask)
