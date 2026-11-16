import triton
import triton.language as tl

@triton.jit
def fused_layernorm_rowwise(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    sum_x = tl.sum(x, 0)
    mean = tl.div_rn(sum_x, N)
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, 0) / N
    inv_std = tl.rsqrt(var + 1e-5)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    normed = (centered * inv_std) * w + b
    tl.store(Y + row * N + cols, normed, mask=mask)
