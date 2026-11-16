import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    off = row * N + col
    mask = col < N

    # Load input elements or zero for out-of-bound lanes
    x = tl.load(X + off, mask, other=0.0)

    # Compute mean: sum and divide by N
    sum_x = tl.sum(x, 0)
    mean = tl.div_rn(sum_x, N)

    # Centered x
    centered = tl.where(mask, x - mean, 0.0)

    # Compute variance: sum of squared centered values
    sum_var = tl.sum(centered * centered, 0)
    invvar = tl.rsqrt(tl.div_rn(sum_var, N) + 1e-5)

    # Load weights and bias with masking
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)

    # Apply layernorm: normalized * w + b
    y = (centered * invvar) * w + b

    # Store output with masking
    tl.store(Y + off, y, mask)
