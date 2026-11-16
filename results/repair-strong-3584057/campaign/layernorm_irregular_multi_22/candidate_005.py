import triton
import triton.language as tl

# First stage: each row is split into 2 tiles of BLOCK=512 (power of two)
@triton.jit
def partial_layernorm_rowwise(X, S, N: tl.constexpr, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK
    cols = tl.arange(0, BLOCK) + offs
    mask = cols < N
    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    part_sum = tl.sum(x, 0)
    part_sumsq = tl.sum(x * x, 0)
    # S shape: [ROWS, NSPLIT, 2] (sum, sumsq per split)
    s_ptr = S + row * NSPLIT * 2 + split * 2
    tl.store(s_ptr + 0, part_sum)
    tl.store(s_ptr + 1, part_sumsq)

# Second stage: reduce splits, normalize, affine-transform
@triton.jit
def final_layernorm_rowwise(S, W, B, X, Y, N: tl.constexpr, BLOCK: tl.constexpr, NSPLIT: tl.constexpr):
    row = tl.program_id(0)
    # Accumulate partials
    total_sum = 0.0
    total_sumsq = 0.0
    for split in range(NSPLIT):
        s_ptr = S + row * NSPLIT * 2 + split * 2
        total_sum += tl.load(s_ptr + 0)
        total_sumsq += tl.load(s_ptr + 1)
    mean = tl.div_rn(total_sum, N)
    var = tl.div_rn(total_sumsq, N) - mean * mean
    inv_std = tl.rsqrt(var + 1e-5)
    # Process entire row in BLOCK-sized tiles
    n_tiles = (N + BLOCK - 1) // BLOCK
    for tile in range(n_tiles):
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
