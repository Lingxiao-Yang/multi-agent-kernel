import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_multi_stage(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, col_mask, other=-float('inf'))
    # Four sub-blocks: indices for each quarter
    quarter = BLOCK // 4
    idx0 = tl.arange(0, quarter)
    idx1 = tl.arange(quarter, 2 * quarter)
    idx2 = tl.arange(2 * quarter, 3 * quarter)
    idx3 = tl.arange(3 * quarter, BLOCK)
    mask0 = idx0 < BLOCK
    mask1 = idx1 < BLOCK
    mask2 = idx2 < BLOCK
    mask3 = idx3 < BLOCK
    # Max reductions in sub-blocks
    mx0 = tl.max(x[idx0] * mask0, 0)
    mx1 = tl.max(x[idx1] * mask1, 0)
    mx2 = tl.max(x[idx2] * mask2, 0)
    mx3 = tl.max(x[idx3] * mask3, 0)
    mx_local = tl.maximum(tl.maximum(mx0, mx1), tl.maximum(mx2, mx3))
    # Exponential and sum reductions in sub-blocks
    exp_x = tl.exp(x - mx_local) * col_mask
    sum0 = tl.sum(exp_x[idx0] * mask0, 0)
    sum1 = tl.sum(exp_x[idx1] * mask1, 0)
    sum2 = tl.sum(exp_x[idx2] * mask2, 0)
    sum3 = tl.sum(exp_x[idx3] * mask3, 0)
    denom = sum0 + sum1 + sum2 + sum3
    denom = tl.where(denom == 0, 1, denom)
    y = exp_x / denom
    tl.store(Y + row * N + offs, y, col_mask)
