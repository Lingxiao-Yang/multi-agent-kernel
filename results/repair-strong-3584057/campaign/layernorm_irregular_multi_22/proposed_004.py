import triton
import triton.language as tl

# First stage: compute partial sum and sumsq for each row-split, store to S
@triton.jit
def partial_layernorm_rowwise(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK
    cols = tl.arange(0, BLOCK) + offs
    mask = cols < N
    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    part_sum = tl.sum(x, 0)
    part_sumsq = tl.sum(x * x, 0)
    # S shape: [ROWS, 2, 2] (2 splits, sum/sumsq)
    s_ptr = S + row * 4 + split * 2
    tl.store(s_ptr + 0, part_sum)
    tl.store(s_ptr + 1, part_sumsq)

# Second stage: load partials, compute mean/var, normalize, affine, store
@triton.jit
def final_layernorm_rowwise(S, W, B, X, Y, N: tl.constexpr, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    row = tl.program_id(0)
    # accumulate partial sums across splits
    total_sum = 0.0
    total_sumsq = 0.0
    for split in range(NSPLIT):
        s_ptr = S + row * 4 + split * 2
        total_sum += tl.load(s_ptr + 0)
        total_sumsq += tl.load(s_ptr + 1)
    mean = tl.div_rn(total_sum, N)
    var = tl.div_rn(total_sumsq, N) - mean * mean
    inv_std = tl.rsqrt(var + 1e-5)
    # process entire row in BLOCK-sized tiles
    for tile in range((N + BLOCK - 1) // BLOCK):
        offs = tile * BLOCK
        cols = tl.arange(0, BLOCK) + offs
        mask = cols < N
        x_ptrs = X + row * N + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        w = tl.load(W + cols, mask=mask, other=0.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        normed = (centered * inv_std) * w + b
        y_ptrs = Y + row * N + cols
        tl.store(y_ptrs, normed, mask=mask)
