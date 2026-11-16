import triton
import triton.language as tl

@triton.jit
def layernorm_fused_row_parallel(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    offs = row * N + cols
    mask = cols < N
    # Load input row with padding masked
    x = tl.load(X + offs, mask=mask, other=0.0)
    # Compute row mean (masked reduction)
    sum_x = tl.sum(x, axis=0)
    mean = tl.div_rn(sum_x, N)
    # Center and mask again for variance (to avoid padding contribution)
    x_centered = x - mean
    x_centered = tl.where(mask, x_centered, 0.0)
    sum_var = tl.sum(x_centered * x_centered, axis=0)
    var = tl.div_rn(sum_var, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # Load weight and bias
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    # Final layernorm: (x - mean) / sqrt(var + eps) * w + b
    y = (x_centered * inv_std) * w + b
    tl.store(Y + offs, y, mask=mask)
