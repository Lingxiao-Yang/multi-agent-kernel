import triton
import triton.language as tl

# First stage: Each row is split into NSPLIT tiles of BLOCK=512. Only valid logical columns are processed.
@triton.jit
def partial_layernorm_rowwise_split(X, S, N: tl.constexpr, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    start = split * BLOCK
    # Compute valid cols for this split: covers [start, min(start+BLOCK,N))
    cols = tl.arange(0, BLOCK) + start
    mask = cols < N
    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    part_sum = tl.sum(tl.where(mask, x, 0.0), 0)
    part_sumsq = tl.sum(tl.where(mask, x * x, 0.0), 0)
    # Store [sum, sumsq] for (row, split) at offset (row * NSPLIT + split) * 2
    s_offset = (row * NSPLIT + split) * 2
    tl.store(S + s_offset + 0, part_sum)
    tl.store(S + s_offset + 1, part_sumsq)

# Second stage: reduce splits, compute mean/var, normalize, affine-transform
@triton.jit
def final_layernorm_rowwise_split(S, W, B, X, Y, N: tl.constexpr, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    row = tl.program_id(0)
    total_sum = 0.0
    total_sumsq = 0.0
    for split in range(NSPLIT):
        s_offset = (row * NSPLIT + split) * 2
        total_sum += tl.load(S + s_offset + 0)
        total_sumsq += tl.load(S + s_offset + 1)
    mean = tl.div_rn(total_sum, N)
    var = tl.div_rn(total_sumsq, N) - mean * mean
    inv_std = tl.rsqrt(var + 1e-5)
    n_tiles = (N + BLOCK - 1) // BLOCK
    for tile in range(n_tiles):
        start = tile * BLOCK
        cols = tl.arange(0, BLOCK) + start
        mask = cols < N
        x_ptrs = X + row * N + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        w = tl.load(W + cols, mask=mask, other=0.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        normed = (centered * inv_std) * w + b
        y_ptrs = Y + row * N + cols
        tl.store(y_ptrs, normed, mask=mask)
