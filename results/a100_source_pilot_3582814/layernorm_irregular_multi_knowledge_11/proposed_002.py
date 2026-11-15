import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sum_val = 0.0
    sq_sum_val = 0.0
    # Tile loop for mean and variance computation
    for tile_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = tile_start + offs
        mask = col < N
        ptr = X + row * N + col
        x = tl.load(ptr, mask, other=0.0)
        sum_val += tl.sum(x, mask)
    mean = sum_val / N

    # Barrier for synchronization across stages
    tl.barrier()

    for tile_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = tile_start + offs
        mask = col < N
        ptr = X + row * N + col
        x = tl.load(ptr, mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        sq_sum_val += tl.sum(centered * centered, mask)
    var = sq_sum_val / N
    inv_std = tl.rsqrt(var + 1e-5)

    # Barrier to ensure mean and inv_std ready
    tl.barrier()

    # Compute output in tiles
    for tile_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = tile_start + offs
        mask = col < N
        x_ptr = X + row * N + col
        x = tl.load(x_ptr, mask, other=0.0)
        w = tl.load(W + col, mask, other=0.0)
        b = tl.load(B + col, mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        y = (centered * inv_std) * w + b
        y_ptr = Y + row * N + col
        tl.store(y_ptr, y, mask)
