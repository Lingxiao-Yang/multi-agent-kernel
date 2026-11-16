import triton
import triton.language as tl

@triton.jit
def layernorm_fused_unrolled_indexed_8warps(X, W, B, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N
    idx = row * N + offs

    # Load input with masking for padding
    x = tl.load(X + idx, mask, other=0.0)

    # Zero out invalid lanes before summation
    x_masked = tl.where(mask, x, 0.0)

    # Manual unrolled sum reduction over 4 segments using indexed offset loads
    step = RB // 4
    offs0 = tl.arange(0, step)
    offs1 = offs0 + step
    offs2 = offs0 + 2 * step
    offs3 = offs0 + 3 * step

    sum0 = tl.sum(tl.where(offs0 < N, x_masked[offs0], 0.0), 0)
    sum0 += tl.sum(tl.where(offs1 < N, x_masked[offs1], 0.0), 0)
    sum0 += tl.sum(tl.where(offs2 < N, x_masked[offs2], 0.0), 0)
    sum0 += tl.sum(tl.where(offs3 < N, x_masked[offs3], 0.0), 0)

    mean = tl.div_rn(sum0, N)

    # Centered input
    centered = tl.where(mask, x - mean, 0.0)

    # Zero out invalid lanes before variance summation
    centered_masked = tl.where(mask, centered, 0.0)

    # Manual unrolled variance reduction over same 4 segments
    sum2 = tl.sum(tl.where(offs0 < N, centered_masked[offs0] * centered_masked[offs0], 0.0), 0)
    sum2 += tl.sum(tl.where(offs1 < N, centered_masked[offs1] * centered_masked[offs1], 0.0), 0)
    sum2 += tl.sum(tl.where(offs2 < N, centered_masked[offs2] * centered_masked[offs2], 0.0), 0)
    sum2 += tl.sum(tl.where(offs3 < N, centered_masked[offs3] * centered_masked[offs3], 0.0), 0)

    var = tl.div_rn(sum2, N)

    # Compute inverse stddev with epsilon
    inv_std = tl.rsqrt(var + 1e-5)

    # Load affine parameters with masking
    w = tl.load(W + offs, mask, other=0.0)
    b = tl.load(B + offs, mask, other=0.0)

    # Normalize and apply affine transform
    y = tl.where(mask, centered * inv_std * w + b, 0.0)

    # Store result with masking
    tl.store(Y + idx, y, mask)
