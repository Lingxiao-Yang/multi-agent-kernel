import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_parallel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, col_mask, other=-float('inf'))
    seg = 4
    seg_size = BLOCK // seg
    # Compute max for each segment using scalar accumulation
    offs_full = offs
    max_val0 = tl.max(tl.where((offs_full < seg_size) & col_mask, x, -float('inf')), 0)
    max_val1 = tl.max(tl.where((offs_full >= seg_size) & (offs_full < 2 * seg_size) & col_mask, x, -float('inf')), 0)
    max_val2 = tl.max(tl.where((offs_full >= 2 * seg_size) & (offs_full < 3 * seg_size) & col_mask, x, -float('inf')), 0)
    max_val3 = tl.max(tl.where((offs_full >= 3 * seg_size) & (offs_full < BLOCK) & col_mask, x, -float('inf')), 0)
    mx = max_val0
    mx = tl.maximum(mx, max_val1)
    mx = tl.maximum(mx, max_val2)
    mx = tl.maximum(mx, max_val3)
    exp_x = tl.exp(x - mx) * col_mask
    # Compute sum for each segment using scalar accumulation
    sum_val0 = tl.sum(tl.where((offs_full < seg_size) & col_mask, exp_x, 0.0), 0)
    sum_val1 = tl.sum(tl.where((offs_full >= seg_size) & (offs_full < 2 * seg_size) & col_mask, exp_x, 0.0), 0)
    sum_val2 = tl.sum(tl.where((offs_full >= 2 * seg_size) & (offs_full < 3 * seg_size) & col_mask, exp_x, 0.0), 0)
    sum_val3 = tl.sum(tl.where((offs_full >= 3 * seg_size) & (offs_full < BLOCK) & col_mask, exp_x, 0.0), 0)
    denom = sum_val0 + sum_val1 + sum_val2 + sum_val3
    denom = tl.where(denom == 0, 1, denom)
    y = exp_x / denom
    tl.store(Y + row * N + offs, y, col_mask)
