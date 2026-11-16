import triton
import triton.language as tl

@triton.jit
def softmax_split_col(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    tile = BLOCK // 2
    offs = tl.arange(0, tile)
    start = part * tile
    cols = start + offs
    col_mask = cols < N
    # load x for this tile
    x_ptr = X + row * N + cols
    x = tl.load(x_ptr, col_mask, other=-float('inf'))
    # local tile max
    mx_local = tl.max(x, 0)
    # manually exchange max with other part (simulate register exchange for 2 parts)
    # load the other tile's x
    other_start = (1 - part) * tile
    other_cols = other_start + offs
    other_mask = other_cols < N
    x_ptr_other = X + row * N + other_cols
    x_other = tl.load(x_ptr_other, other_mask, other=-float('inf'))
    mx_other = tl.max(x_other, 0)
    mx_total = tl.maximum(mx_local, mx_other)
    # shift and exp
    x_shifted = x - mx_total
    exp_x = tl.exp(x_shifted)
    # local tile sum
    local_sum = tl.sum(exp_x, 0)
    # compute sum for the other tile using the same shift
    x_other_shifted = x_other - mx_total
    exp_x_other = tl.exp(x_other_shifted)
    other_sum = tl.sum(exp_x_other, 0)
    denom = local_sum + other_sum
    # softmax
    softmax = exp_x / denom
    y_ptr = Y + row * N + cols
    tl.store(y_ptr, softmax, col_mask)
