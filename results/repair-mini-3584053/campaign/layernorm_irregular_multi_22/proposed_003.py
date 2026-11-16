import triton
import triton.language as tl

@triton.jit
def layernorm_fused_pipelined(X, W, B, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    mask = offs < N
    idx = row * N + offs

    # Stage 0: Load input with masking and zero padding invalid lanes
    x = tl.load(X + idx, mask, other=0.0)
    x_masked = tl.where(mask, x, 0.0)

    # Pipeline stage 0: compute sum for mean
    s = tl.sum(x_masked, 0)
    mean = tl.div_rn(s, N)

    # Schedule pipeline boundary
    tl.next_stage()

    # Stage 1: compute variance
    centered = tl.where(mask, x - mean, 0.0)
    centered_masked = tl.where(mask, centered, 0.0)
    s2 = tl.sum(centered_masked * centered_masked, 0)
    var = tl.div_rn(s2, N)
    inv_std = tl.rsqrt(var + 1e-5)

    # Load affine parameters with masking
    w = tl.load(W + offs, mask, other=0.0)
    b = tl.load(B + offs, mask, other=0.0)

    # Normalize and affine transform
    y = tl.where(mask, centered * inv_std * w + b, 0.0)

    # Store output with masking
    tl.store(Y + idx, y, mask)
