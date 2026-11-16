import triton
import triton.language as tl

@triton.jit
def fused_softmax_rowblock_parallel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    n_tiles = 2
    tile_cols = (N + n_tiles - 1) // n_tiles
    col_start = tile * tile_cols
    offs = tl.arange(0, BLOCK)
    col = col_start + offs
    mask = col < N
    # --- Compute rowwise max (scalar), parallelized over BLOCK-sized strides ---
    max_val = -float('inf')
    # Loop over the row in BLOCK-sized chunks
    for chunk_start in range(0, N, BLOCK):
        chunk_col = chunk_start + offs
        chunk_mask = chunk_col < N
        x_ptrs = X + row * N + chunk_col
        x = tl.load(x_ptrs, mask=chunk_mask, other=-float('inf'))
        local_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, local_max)
    # --- Compute denominator (scalar), parallelized over BLOCK-sized strides ---
    denom = 0.0
    for chunk_start in range(0, N, BLOCK):
        chunk_col = chunk_start + offs
        chunk_mask = chunk_col < N
        x_ptrs = X + row * N + chunk_col
        x = tl.load(x_ptrs, mask=chunk_mask, other=-float('inf'))
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
