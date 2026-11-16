import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_multi_stage(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, col_mask, other=-float('inf'))
    # Stage 1: Parallel sub-block reduction for max
    quarter = BLOCK // 4
    mx0 = tl.max(x[:quarter], 0)
    mx1 = tl.max(x[quarter:2*quarter], 0)
    mx2 = tl.max(x[2*quarter:3*quarter], 0)
    mx3 = tl.max(x[3*quarter:], 0)
    mx_local = tl.maximum(tl.maximum(mx0, mx1), tl.maximum(mx2, mx3))
    # Stage 2: Compute exp and parallel sum in sub-blocks
    exp_x = tl.exp(x - mx_local) * col_mask
    sum0 = tl.sum(exp_x[:quarter], 0)
    sum1 = tl.sum(exp_x[quarter:2*quarter], 0)
    sum2 = tl.sum(exp_x[2*quarter:3*quarter], 0)
    sum3 = tl.sum(exp_x[3*quarter:], 0)
    denom = sum0 + sum1 + sum2 + sum3
    denom = tl.where(denom == 0, 1, denom)
    y = exp_x / denom
    tl.store(Y + row * N + offs, y, col_mask)
