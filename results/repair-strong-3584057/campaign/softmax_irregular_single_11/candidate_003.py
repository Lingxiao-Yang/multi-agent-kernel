import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_vectorized_loop_fixed(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    vec_width = 4
    max_val = -float('inf')
    # Loop over row in vector chunks to compute row max
    for start in range(0, RB, vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_v, 0))
    # Second pass: compute sum of exp(x - max_val)
    sum_exp = 0.0
    for start in range(0, RB, vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    denom = tl.where(sum_exp == 0, 1, sum_exp)
    # Third pass: normalize and store
    for start in range(0, RB, vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        res = exp_x / denom
        tl.store(Y + idx_v, res, mask=mask_v)
