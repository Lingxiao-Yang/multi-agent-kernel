import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs

    idx = row * N + col
    mask = col < N

    # Load input with masking
    x = tl.load(X + idx, mask, other=0.0)

    # Compute mean using masked sum (mask as positional argument)
    sum_x = tl.sum(x, mask)
    mean = tl.div_rn(sum_x, N)

    # Center input and compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    sum_var = tl.sum(x_centered * x_centered, mask)
    var = tl.div_rn(sum_var, N)
    inv_std = tl.rsqrt(var + 1e-5)

    # Load weights and bias with masking
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)

    # Compute normalized output
    y = tl.where(mask, x_centered * inv_std * w + b, 0.0)

    # Store results with masking
    tl.store(Y + idx, y, mask)
