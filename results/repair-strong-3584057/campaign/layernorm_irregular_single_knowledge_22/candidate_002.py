import triton
import triton.language as tl

@triton.jit
def layernorm_parallel_tile(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    num_tiles = tl.num_programs(1)
    tile_size = (N + num_tiles - 1) // num_tiles
    col_start = tile * tile_size
    col_end = tl.minimum(col_start + tile_size, N)
    # Local accumulators
    local_mean = 0.0
    local_var = 0.0
    count = 0.0
    # 1. Partial mean
    for off in range(col_start, col_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = (offs < col_end) & (offs < N)
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        local_mean += tl.sum(x, 0)
        count += tl.sum(mask, 0)
    # Global reduction: sum over tiles
    total_mean = tl.sum(local_mean, axis=0)
    total_count = tl.sum(count, axis=0)
    mean = total_mean / N
    # 2. Partial variance
    for off in range(col_start, col_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = (offs < col_end) & (offs < N)
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        local_var += tl.sum(centered * centered, 0)
    total_var = tl.sum(local_var, axis=0)
    var = total_var / N
    inv_std = tl.rsqrt(var + 1e-5)
    # 3. Write output for assigned tile
    for off in range(col_start, col_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = (offs < col_end) & (offs < N)
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        w = tl.load(W + offs, mask=mask, other=0.0)
        b = tl.load(B + offs, mask=mask, other=0.0)
        y = (centered * inv_std) * w + b
        y_ptrs = Y + row * N + offs
        tl.store(y_ptrs, y, mask)
