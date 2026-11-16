import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_vectorized_loop(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    idx = row * N + offs
    mask = offs < N
    # Load row in vectorized chunks, accumulate max
    vec_width = 4
    max_val = -float('inf')
    for i in range(0, RB, vec_width):
        offs_v = offs[i:i+vec_width]
        idx_v = row * N + offs_v
        mask_v = offs_v < N
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_v, 0))
    # Compute exp and sum in a second vectorized loop
    sum_exp = 0.0
    for i in range(0, RB, vec_width):
        offs_v = offs[i:i+vec_width]
        idx_v = row * N + offs_v
        mask_v = offs_v < N
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        sum_exp = sum_exp + tl.sum(exp_x, 0)
    denom = tl.where(sum_exp == 0, 1, sum_exp)
    # Final normalization and store in vectorized chunks
    for i in range(0, RB, vec_width):
        offs_v = offs[i:i+vec_width]
        idx_v = row * N + offs_v
        mask_v = offs_v < N
        x_v = tl.load(X + idx_v, mask=mask_v, other=-float('inf'))
        x_centered = x_v - max_val
        exp_x = tl.exp(x_centered) * mask_v
        res = exp_x / denom
        tl.store(Y + idx_v, res, mask=mask_v)
