import triton
import triton.language as tl

@triton.jit
def softmax_split_col_fixed(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    tile = RB // 2
    offs = tl.arange(0, tile)
    start = part * tile
    cols = start + offs
    col_mask = cols < N
    x_ptr = X + row * N + cols
    x = tl.load(x_ptr, col_mask, other=-float('inf'))
    mx_local = tl.max(x, 0)
    # compute global max by also loading other tile
    other_start = (1 - part) * tile
    other_cols = other_start + offs
    other_col_mask = other_cols < N
    x_ptr_other = X + row * N + other_cols
    x_other = tl.load(x_ptr_other, other_col_mask, other=-float('inf'))
    mx_other = tl.max(x_other, 0)
    mx_total = tl.maximum(mx_local, mx_other)
    x_shifted = x - mx_total
    exp_x = tl.exp(x_shifted)
    local_sum = tl.sum(exp_x, 0)
    x_other_shifted = x_other - mx_total
    exp_x_other = tl.exp(x_other_shifted)
    other_sum = tl.sum(exp_x_other, 0)
    denom = local_sum + other_sum
    softmax = exp_x / denom
    y_ptr = Y + row * N + cols
    tl.store(y_ptr, softmax, col_mask)
