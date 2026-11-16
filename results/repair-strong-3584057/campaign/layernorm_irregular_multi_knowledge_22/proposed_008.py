import triton
import triton.language as tl

# First pass: each program covers a BLOCK=512 segment of a row, computes partial sums/sumsq (for two splits), writes to S
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
    # Each row gets 4 floats in S: [sum0, sumsq0, sum1, sumsq1]
    tl.store(S + row_id * 4 + split * 2 + 0, part_sum)
    tl.store(S + row_id * 4 + split * 2 + 1, part_sumsq)

# Second pass: each program covers a BLOCK=512 segment of a row, aggregates statistics from S, normalizes, writes Y
@triton.jit
def layernorm_blockwise_final_ws_fullrow(X, W, B, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # Load statistics from S
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
    # Normalize over this segment of the row
    x_ptr = X + row_id * N + offs
    x = tl.load(x_ptr, mask, other=0)
    w = tl.load(W + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    norm = (x - mean) * inv_std
    y = norm * w + b
    # Always write to Y with masking for out-of-bounds columns
    tl.store(Y + row_id * N + offs, tl.where(mask, y, 0.0))
