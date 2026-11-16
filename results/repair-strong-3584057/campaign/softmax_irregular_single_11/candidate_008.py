import triton
import triton.language as tl

@triton.jit
def softmax_fused_vectorized_double_buffer_fixed_v3(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    vec_width = 8
    # Compute row max: process all even and odd chunks in one loop (stride 8)
    max_val = -float('inf')
    for start in tl.static_range(0, RB, vec_width):
        offs_v = start + tl.arange(0, vec_width, 1)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        mx_v = tl.max(x_v, 0)
        max_val = tl.maximum(max_val, mx_v)
    # Sum of exp(x - max)
    sum_exp = 0.0
    for start in tl.static_range(0, RB, vec_width):
        offs_v = start + tl.arange(0, vec_width, 1)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    denom = tl.where(sum_exp == 0, 1.0, sum_exp)
    inv_denom = 1.0 / denom
    # Normalize and store
    for start in tl.static_range(0, RB, vec_width):
        offs_v = start + tl.arange(0, vec_width, 1)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        res = exp_x * inv_denom
        tl.store(Y + idx_v, res, mask=mask_v)
