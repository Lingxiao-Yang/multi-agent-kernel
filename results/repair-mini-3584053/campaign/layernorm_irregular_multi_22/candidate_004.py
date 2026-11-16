import triton
import triton.language as tl

@triton.jit
def layernorm_fused_unrolled_8warps(X, W, B, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N
    idx = row * N + offs

    # Load input with masking for padding
    x = tl.load(X + idx, mask, other=0.0)

    # Zero out invalid lanes before summation (no mask argument for tl.sum)
    x_masked = tl.where(mask, x, 0.0)

    # Manual unrolled reduction for sum (mean)
    step = RB // 4
    sum0 = tl.sum(x_masked[0:step], 0)
    sum0 += tl.sum(x_masked[step:2*step], 0)
    sum0 += tl.sum(x_masked[2*step:3*step], 0)
    sum0 += tl.sum(x_masked[3*step:], 0)
    mean = tl.div_rn(sum0, N)

    # Centered input
    centered = tl.where(mask, x - mean, 0.0)

    # Zero out invalid lanes before variance summation
    centered_masked = tl.where(mask, centered, 0.0)

    # Manual unrolled reduction for variance
    sum2 = tl.sum(centered_masked[0:step] * centered_masked[0:step], 0)
    sum2 += tl.sum(centered_masked[step:2*step] * centered_masked[step:2*step], 0)
    sum2 += tl.sum(centered_masked[2*step:3*step] * centered_masked[2*step:3*step], 0)
    sum2 += tl.sum(centered_masked[3*step:] * centered_masked[3*step:], 0)
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
