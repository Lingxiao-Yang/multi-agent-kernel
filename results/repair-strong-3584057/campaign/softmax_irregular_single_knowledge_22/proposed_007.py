import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_unrolled8(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, col_mask, other=-float('inf'))
    # Unroll the reduction into 8 segments to increase instruction-level parallelism
    seg = 8
    seg_size = BLOCK // seg
    # Compute max for each segment
    max_vals = []
    for i in range(seg):
        seg_start = i * seg_size
        seg_end = (i + 1) * seg_size if i < seg - 1 else BLOCK
        seg_mask = (offs >= seg_start) & (offs < seg_end) & col_mask
        max_i = tl.max(tl.where(seg_mask, x, -float('inf')), 0)
        max_vals.append(max_i)
    mx = max_vals[0]
    for i in range(1, seg):
        mx = tl.maximum(mx, max_vals[i])
    # Compute exponentials
    exp_x = tl.exp(x - mx) * col_mask
    # Compute sum for each segment
    sum_vals = []
    for i in range(seg):
        seg_start = i * seg_size
        seg_end = (i + 1) * seg_size if i < seg - 1 else BLOCK
        seg_mask = (offs >= seg_start) & (offs < seg_end) & col_mask
        sum_i = tl.sum(tl.where(seg_mask, exp_x, 0.0), 0)
        sum_vals.append(sum_i)
    denom = sum_vals[0]
    for i in range(1, seg):
        denom = denom + sum_vals[i]
    denom = tl.where(denom == 0, 1, denom)
    y = exp_x / denom
    tl.store(Y + row * N + offs, y, col_mask)
