import triton
import triton.language as tl

@triton.jit
def layernorm_fused_unrolled_constexpr_8warps(X, W, B, Y, N: tl.constexpr, RB: tl.constexpr, step: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N
    idx = row * N + offs

    # Load input with masking for padding
    x = tl.load(X + idx, mask, other=0.0)

    # Define explicit constexpr offset arrays for manual unrolling
    offs0 = tl.arange(0, step)
    offs1 = offs0 + step
    offs2 = offs0 + 2 * step
    offs3 = offs0 + 3 * step

    # Partial sums with masking for mean
    sum0 = tl.sum(tl.where(offs0 < N, tl.load(X + row * N + offs0, offs0 < N, other=0.0), 0.0), 0)
    sum0 += tl.sum(tl.where(offs1 < N, tl.load(X + row * N + offs1, offs1 < N, other=0.0), 0.0), 0)
    sum0 += tl.sum(tl.where(offs2 < N, tl.load(X + row * N + offs2, offs2 < N, other=0.0), 0.0), 0)
    sum0 += tl.sum(tl.where(offs3 < N, tl.load(X + row * N + offs3, offs3 < N, other=0.0), 0.0), 0)

    mean = tl.div_rn(sum0, N)

    # Centered input with masking
    centered = tl.where(mask, x - mean, 0.0)

    # Partial sums for variance
    sum2 = tl.sum(tl.where(offs0 < N, centered[offs0] * centered[offs0], 0.0), 0)
    sum2 += tl.sum(tl.where(offs1 < N, centered[offs1] * centered[offs1], 0.0), 0)
    sum2 += tl.sum(tl.where(offs2 < N, centered[offs2] * centered[offs2], 0.0), 0)
    sum2 += tl.sum(tl.where(offs3 < N, centered[offs3] * centered[offs3], 0.0), 0)

    var = tl.div_rn(sum2, N)

    inv_std = tl.rsqrt(var + 1e-5)

    # Load affine parameters with masking
    w = tl.load(W + offs, mask, other=0.0)
    b = tl.load(B + offs, mask, other=0.0)

    y = tl.where(mask, centered * inv_std * w + b, 0.0)

    tl.store(Y + idx, y, mask)
