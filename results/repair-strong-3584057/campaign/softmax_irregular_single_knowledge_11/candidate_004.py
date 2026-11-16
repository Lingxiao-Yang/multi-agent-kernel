import triton
import triton.language as tl

@triton.jit
def fused_softmax_tile(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    num_tiles = 2
    tile_cols = (N + num_tiles - 1) // num_tiles
    col_start = tile_id * tile_cols
    col = col_start + offs
    mask = col < N
    # --- Row-wise max: all tiles for this row must agree ---
    max_val = -float('inf')
    for t in range(num_tiles):
        blk_col_start = t * tile_cols
        blk_col = blk_col_start + offs
        blk_mask = blk_col < N
        x_ptrs = X + row * N + blk_col
        x = tl.load(x_ptrs, mask=blk_mask, other=-float('inf'))
        tile_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, tile_max)
    max_val = tl.max(max_val, 0)
    # --- Denominator: sum(exp(x - max)) ---
    denom = 0.0
    for t in range(num_tiles):
        blk_col_start = t * tile_cols
        blk_col = blk_col_start + offs
        blk_mask = blk_col < N
        x_ptrs = X + row * N + blk_col
        x = tl.load(x_ptrs, mask=blk_mask, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        tile_sum = tl.sum(exp_x, 0)
        denom = denom + tile_sum
    # --- Softmax for this tile ---
    x_ptrs = X + row * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    exp_x = tl.exp(x - max_val)
    softmax = exp_x / denom
    y_ptrs = Y + row * N + col
    tl.store(y_ptrs, softmax, mask=mask)
