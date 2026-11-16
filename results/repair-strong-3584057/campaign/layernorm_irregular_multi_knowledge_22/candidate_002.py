import triton
import triton.language as tl

@triton.jit
def layernorm_multiwarp(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x_ptr = X + row_id * N + offs
    mask = offs < N

    # Load input row, padding with 0 where out-of-bounds
    x = tl.load(x_ptr, mask, other=0)
    # Multi-warp: all threads in block participate in reduction
    sum_x = tl.sum(x, 0)
    mean = tl.div_rn(sum_x, N)
    centered = tl.where(mask, x - mean, 0)
    sum_x2 = tl.sum(centered * centered, 0)
    var = tl.div_rn(sum_x2, N)
    inv_std = tl.rsqrt(var + 1e-5)

    # Reload input and parameters
    x = tl.load(x_ptr, mask, other=0)
    w = tl.load(W + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    norm = (x - mean) * inv_std
    y = norm * w + b
    tl.store(Y + row_id * N + offs, y, mask)
