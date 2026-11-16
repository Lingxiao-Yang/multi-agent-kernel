import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr):
    row = tl.program_id(0)
    BLOCK = 256  # power of two <= N
    offs = tl.arange(0, BLOCK)
    col = offs
    mask = col < N

    # Load input segment with masking
    x = tl.load(X + row * N + col, mask, other=0.0)

    # Compute mean: sum reduction with masking
    s = tl.sum(tl.where(mask, x, 0.0), 0)
    mean = s / N

    # Compute variance: sum of squared deviations
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, 0) / N
    inv_std = tl.rsqrt(var + 1e-5)

    # Load weight and bias with masking
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)

    # Compute normalized output and store with masking
    y = (x - mean) * inv_std * w + b
    tl.store(Y + row * N + col, y, mask)
