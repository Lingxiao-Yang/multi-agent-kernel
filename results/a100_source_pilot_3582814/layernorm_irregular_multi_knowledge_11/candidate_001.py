import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    mask = offs < N

    # Load input row with masking
    x_ptr = X + row * N + offs
    x = tl.load(x_ptr, mask, other=0.0)

    # Compute mean of the row
    mean = tl.sum(x, mask) / N

    # Center the input
    centered = tl.where(mask, x - mean, 0.0)

    # Compute variance and inv stddev
    var = tl.sum(centered * centered, mask) / N
    inv_std = tl.rsqrt(var + 1e-5)

    # Load weight and bias
    w = tl.load(W + offs, mask, other=0.0)
    b = tl.load(B + offs, mask, other=0.0)

    # Apply layernorm fused operation
    y = (centered * inv_std) * w + b

    # Store output with masking
    y_ptr = Y + row * N + offs
    tl.store(y_ptr, y, mask)
