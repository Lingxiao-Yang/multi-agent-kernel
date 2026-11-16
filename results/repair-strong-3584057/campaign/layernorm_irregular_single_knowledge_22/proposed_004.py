import triton
import triton.language as tl

@triton.jit
def layernorm_fused_blockloop_parallel(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    TILES = 4
    tile_size = (N + TILES - 1) // TILES
    col_start = tile * tile_size
    col_end = tl.minimum(col_start + tile_size, N)
    # Partial accumulators
    mean_acc = 0.0
    mean2_acc = 0.0
    count = 0.0
    for off in range(col_start, col_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = (offs < col_end) & (offs < N)
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        mean_acc += tl.sum(x, 0)
        mean2_acc += tl.sum(x * x, 0)
        count += tl.sum(mask, 0)
    # Store to buffer for cross-tile reduction
    mean_buf = tl.zeros((TILES,), dtype=tl.float32)
    mean2_buf = tl.zeros((TILES,), dtype=tl.float32)
    count_buf = tl.zeros((TILES,), dtype=tl.float32)
    mean_buf[tile] = mean_acc
    mean2_buf[tile] = mean2_acc
    count_buf[tile] = count
    # All tiles: sum across all tile slots
    total_mean = tl.sum(mean_buf, 0) / tl.sum(count_buf, 0)
    total_mean2 = tl.sum(mean2_buf, 0) / tl.sum(count_buf, 0)
    var = total_mean2 - total_mean * total_mean
    inv_std = tl.rsqrt(var + 1e-5)
    # Output pass: apply normalization and affine
    for off in range(col_start, col_end, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = (offs < col_end) & (offs < N)
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        centered = tl.where(mask, x - total_mean, 0.0)
        w = tl.load(W + offs, mask=mask, other=0.0)
        b = tl.load(B + offs, mask=mask, other=0.0)
        y = (centered * inv_std) * w + b
        y_ptrs = Y + row * N + offs
        tl.store(y_ptrs, y, mask)
