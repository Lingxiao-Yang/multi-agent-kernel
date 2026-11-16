import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    # Load input with masking for irregular N
    x = tl.load(X + offs, col < N, other=0)
    # Compute mean across the row
    mean = tl.div_rn(tl.sum(x, 0), N)
    # Centered x for variance
    centered = tl.where(col < N, x - mean, 0)
    var = tl.sum(centered * centered, 0) / N
    inv = tl.rsqrt(var + 1e-5)
    # Load weights and bias with masking
    w = tl.load(W + col, col < N, other=0)
    b = tl.load(B + col, col < N, other=0)
    # Apply normalization, scale, shift
    norm = tl.where(col < N, (x - mean) * inv * w + b, 0)
    tl.store(Y + offs, norm, col < N)
