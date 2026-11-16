import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_parallel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptrs = X + row * N + offs
    # Load input with masking
    x = tl.load(x_ptrs, col_mask, other=-float('inf'))
    # Parallel segmented max reduction (4 segments)
    seg = 4
    seg_size = BLOCK // seg
    offs_full = tl.arange(0, BLOCK)
    max_vals = tl.full((seg,), -float('inf'), dtype=tl.float32)
    for i in range(seg):
        seg_mask = (offs_full >= i * seg_size) & (offs_full < (i + 1) * seg_size)
        seg_mask = seg_mask & col_mask
        seg_x = tl.where(seg_mask, x, -float('inf'))
        max_val = tl.max(seg_x, 0)
        max_vals = tl.where(tl.arange(0, seg) == i, max_val, max_vals)
    mx = max_vals[0]
    mx = tl.maximum(mx, max_vals[1])
    mx = tl.maximum(mx, max_vals[2])
    mx = tl.maximum(mx, max_vals[3])
    # Compute exponentials with mask
    exp_x = tl.exp(x - mx) * col_mask
    # Parallel segmented sum reduction (4 segments)
    sum_vals = tl.zeros((seg,), dtype=tl.float32)
    for i in range(seg):
        seg_mask = (offs_full >= i * seg_size) & (offs_full < (i + 1) * seg_size)
        seg_mask = seg_mask & col_mask
        seg_exp = tl.where(seg_mask, exp_x, 0.0)
        sum_val = tl.sum(seg_exp, 0)
        sum_vals = tl.where(tl.arange(0, seg) == i, sum_val, sum_vals)
    denom = sum_vals[0] + sum_vals[1] + sum_vals[2] + sum_vals[3]
    denom = tl.where(denom == 0, 1, denom)
    y = exp_x / denom
    tl.store(Y + row * N + offs, y, col_mask)
