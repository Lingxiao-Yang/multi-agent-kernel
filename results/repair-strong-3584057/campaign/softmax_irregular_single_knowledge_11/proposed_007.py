import triton
import triton.language as tl

@triton.jit
def fused_softmax_rowblock(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    n_tiles = 2
    tile_cols = (N + n_tiles - 1) // n_tiles
    col_start = tile * tile_cols
    offs = tl.arange(0, BLOCK)
    col = col_start + offs
    mask = col < N
    # --- Compute rowwise max across all tiles (scalar max) ---
    max_val = -float('inf')
    for t in range(n_tiles):
        t_col_start = t * tile_cols
        t_col = t_col_start + offs
        t_mask = t_col < N
        x_ptrs = X + row * N + t_col
        x = tl.load(x_ptrs, mask=t_mask, other=-float('inf'))
        local_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, local_max)
    # --- Compute denominator (scalar sum) ---
    denom = 0.0
    for t in range(n_tiles):
        t_col_start = t * tile_cols
        t_col = t_col_start + offs
        t_mask = t_col < N
        x_ptrs = X + row * N + t_col
        x = tl.load(x_ptrs, mask=t_mask, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        local_sum = tl.sum(exp_x, 0)
        denom = denom + local_sum
    # --- Compute and store softmax for this tile ---
    x_ptrs = X + row * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    numerators = tl.exp(x - max_val)
    softmax = numerators / denom
    out_ptrs = Y + row * N + col
    tl.store(out_ptrs, softmax, mask=mask)
