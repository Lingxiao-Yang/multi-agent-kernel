import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs

    idx = row * N + col
    mask = col < N

    # Load input with masking, zero out-of-bounds
    x = tl.load(X + idx, mask, other=0.0)

    # Zero out-of-bounds elements to exclude padding from sums
    x_masked = tl.where(mask, x, 0.0)

    # Compute mean using accurate division
    sum_x = tl.sum(x_masked)
    mean = tl.div_rn(sum_x, N)

    # Center input and zero out-of-bounds for variance
    x_centered = tl.where(mask, x - mean, 0.0)
    sum_var = tl.sum(x_centered * x_centered)
    var = tl.div_rn(sum_var, N)
    inv_std = tl.rsqrt(var + 1e-5)

    # Load weights and biases with masking
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)

    # Compute normalized output with masking
    y = tl.where(mask, x_centered * inv_std * w + b, 0.0)

    # Store results with masking
    tl.store(Y + idx, y, mask)
