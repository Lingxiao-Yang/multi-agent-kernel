import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x_ptr = X + row_id * N + offs
    mask = offs < N

    # Load input row, padding with 0 where out-of-bounds
    x = tl.load(x_ptr, mask, other=0)
    # Compute mean
    mean = tl.div_rn(tl.sum(x, 0), N)
    # Centered vector, mask out-of-bounds
    centered = tl.where(mask, x - mean, 0)
    # Variance: sum of squares divided by N
    var = tl.sum(centered * centered, 0) / N
    inv_std = tl.rsqrt(var + 1e-5)

    # Reload input row for numerics (preserve original x)
    x = tl.load(x_ptr, mask, other=0)
    # Load weight and bias for the row's columns
    w = tl.load(W + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    # Apply normalization only for in-bounds columns
    norm = (x - mean) * inv_std
    y = norm * w + b
    tl.store(Y + row_id * N + offs, y, mask)
