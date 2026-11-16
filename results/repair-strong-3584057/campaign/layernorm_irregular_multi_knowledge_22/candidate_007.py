import triton
import triton.language as tl

# First pass: compute partial row statistics and write them into a dedicated workspace buffer S
@triton.jit
def layernorm_blockwise_partial_ws(X, N: tl.constexpr, BLOCK: tl.constexpr, S):
    row_id = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x_ptr = X + row_id * N + offs
    x = tl.load(x_ptr, mask, other=0)
    part_sum = tl.sum(x, 0)
    part_sumsq = tl.sum(x * x, 0)
    # Write partials into workspace S: shape [513,4], layout [row_id, split*2 + <0,1>]
    tl.store(S + row_id * 4 + split * 2 + 0, part_sum)
    tl.store(S + row_id * 4 + split * 2 + 1, part_sumsq)

# Second pass: aggregate partials from S, compute mean/var, normalize and output
@triton.jit
def layernorm_blockwise_final_ws(X, W, B, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # Read partial sums from workspace S
    sum0 = tl.load(S + row_id * 4 + 0)
    sumsq0 = tl.load(S + row_id * 4 + 1)
    sum1 = tl.load(S + row_id * 4 + 2)
    sumsq1 = tl.load(S + row_id * 4 + 3)
    full_sum = sum0 + sum1
    full_sumsq = sumsq0 + sumsq1
    mean = tl.div_rn(full_sum, N)
    mean_sq = mean * mean
    var = tl.div_rn(full_sumsq, N) - mean_sq
    inv_std = tl.rsqrt(var + 1e-5)
    # Load input row for normalization (from X)
    x_ptr = X + row_id * N + offs
    x = tl.load(x_ptr, offs < N, other=0)
    w = tl.load(W + offs, offs < N, other=0)
    b = tl.load(B + offs, offs < N, other=0)
    norm = (x - mean) * inv_std
    y = norm * w + b
    # Always overwrite the entire output region for this row, zero for out-of-bounds
    mask = offs < N
    out = tl.where(mask, y, 0.0)
    tl.store(Y + row_id * N + offs, out)
