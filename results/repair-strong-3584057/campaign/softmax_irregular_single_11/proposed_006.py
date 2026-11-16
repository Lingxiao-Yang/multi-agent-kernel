import triton
import triton.language as tl

@triton.jit
def softmax_fused_vectorized_double_buffer(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    vec_width = 8
    # Pass 1: even vector chunks for row max
    max_val = -float('inf')
    for start in tl.static_range(0, RB, 2 * vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_v, 0))
    # Pass 2: odd vector chunks for row max
    for start in tl.static_range(vec_width, RB, 2 * vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_v, 0))
    # Pass 3: sum of exp(x - max) for even vector chunks
    sum_exp = 0.0
    for start in tl.static_range(0, RB, 2 * vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    # Pass 4: odd vector chunks for sum
    for start in tl.static_range(vec_width, RB, 2 * vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    denom = tl.where(sum_exp == 0, 1.0, sum_exp)
    inv_denom = 1.0 / denom
    # Pass 5: normalize and store for even vector chunks
    for start in tl.static_range(0, RB, 2 * vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        res = exp_x * inv_denom
        tl.store(Y + idx_v, res, mask=mask_v)
    # Pass 6: odd vector chunks for normalize/store
    for start in tl.static_range(vec_width, RB, 2 * vec_width):
        offs_v = start + tl.arange(0, vec_width)
        mask_v = offs_v < N
        idx_v = row * N + offs_v
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        res = exp_x * inv_denom
        tl.store(Y + idx_v, res, mask=mask_v)
