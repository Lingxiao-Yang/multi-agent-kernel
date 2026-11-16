import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N

    # Load input elements with masking, zero for out-of-bounds
    x = tl.load(X + offs, mask, other=0.0)

    # Compute mean with masked sum
    sum_x = tl.sum(tl.where(mask, x, 0.0), 0)
    mean = sum_x / N

    # Centered x with masking
    centered = tl.where(mask, x - mean, 0.0)

    # Compute variance with masked sum
    sum_sq = tl.sum(centered * centered, 0)
    invvar = tl.rsqrt(sum_sq / N + 1e-5)

    # Load weights and bias with masking
    w = tl.load(W + tl.arange(0, BLOCK), mask, other=0.0)
    b = tl.load(B + tl.arange(0, BLOCK), mask, other=0.0)

    # Apply layernorm
    y = (centered * invvar) * w + b

    # Store output with masking
    tl.store(Y + offs, y, mask)
