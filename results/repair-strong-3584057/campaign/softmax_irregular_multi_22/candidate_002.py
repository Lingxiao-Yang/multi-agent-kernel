import triton
import triton.language as tl

@triton.jit
def softmax_tile_split(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    tile_width = BLOCK // 2
    offs = tl.arange(0, tile_width)
    col_start = part * tile_width
    col_idx = col_start + offs
    col_mask = col_idx < N
    x_ptr = X + row * N + col_idx
    x = tl.load(x_ptr, col_mask, other=-float('inf'))
    mx_tile = tl.max(x, 0)
    x_shifted = x - mx_tile
    exp_x = tl.exp(x_shifted)
    denom_tile = tl.sum(exp_x, 0)
    softmax_tile = exp_x / denom_tile
    y_ptr = Y + row * N + col_idx
    tl.store(y_ptr, softmax_tile, col_mask)
