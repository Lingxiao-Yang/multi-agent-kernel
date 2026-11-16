import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N
    idx = row * N + offs

    # Load input with masking for padding
    x = tl.load(X + idx, mask, other=0.0)

    # Compute mean: sum reduction over RB lanes with masking
    s = tl.sum(x, mask)
    mean = tl.div_rn(s, N)

    # Centered input
    centered = tl.where(mask, x - mean, 0.0)

    # Compute variance
    s2 = tl.sum(centered * centered, mask)
    var = tl.div_rn(s2, N)

    # Compute inverse stddev with epsilon
    inv_std = tl.rsqrt(var + 1e-5)

    # Load affine parameters with masking
    w = tl.load(W + offs, mask, other=0.0)
    b = tl.load(B + offs, mask, other=0.0)

    # Normalize and apply affine transform
    y = tl.where(mask, centered * inv_std * w + b, 0.0)

    # Store result with masking
    tl.store(Y + idx, y, mask)
