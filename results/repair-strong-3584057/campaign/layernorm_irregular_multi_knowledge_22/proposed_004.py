import triton
import triton.language as tl

# First pass: compute partial row statistics and write them into the output buffer's padding
@triton.jit
def layernorm_blockwise_partial(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x_ptr = X + row_id * N + offs
    x = tl.load(x_ptr, mask, other=0)
    part_sum = tl.sum(x, 0)
    part_sumsq = tl.sum(x * x, 0)
    # Write partials into the end of the output buffer for this row, after N
    if split == 0:
        tl.store(Y + row_id * N + N, part_sum)
        tl.store(Y + row_id * N + N + 1, part_sumsq)
    else:
        tl.store(Y + row_id * N + N + 2, part_sum)
        tl.store(Y + row_id * N + N + 3, part_sumsq)
    # No normalization or output yet

# Second pass: aggregate partials, compute mean/var, normalize and output
@triton.jit
def layernorm_blockwise_final(Y, W, B, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    # Read partial sums from the end of the buffer
    sum0 = tl.load(Y + row_id * N + N)
    sumsq0 = tl.load(Y + row_id * N + N + 1)
    sum1 = tl.load(Y + row_id * N + N + 2)
    sumsq1 = tl.load(Y + row_id * N + N + 3)
    full_sum = sum0 + sum1
    full_sumsq = sumsq0 + sumsq1
    mean = tl.div_rn(full_sum, N)
    mean_sq = mean * mean
    var = tl.div_rn(full_sumsq, N) - mean_sq
    inv_std = tl.rsqrt(var + 1e-5)
    # Reload input row from original buffer (Y is still uninitialized for output region)
    # For simplicity, expect that X and Y do not overlap, so reload X from Y's start
    x = tl.load(Y + row_id * N + offs, mask, other=0)
    w = tl.load(W + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    norm = (x - mean) * inv_std
    y = norm * w + b
    tl.store(Y + row_id * N + offs, y, mask)
