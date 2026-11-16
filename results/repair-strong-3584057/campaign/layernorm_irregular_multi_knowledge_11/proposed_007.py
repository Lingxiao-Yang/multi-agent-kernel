import triton
import triton.language as tl

@triton.jit
def layernorm_rowwise_splitmean(X, W, B, Y, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    num_tiles = (N + TILE - 1) // TILE
    # First pass: mean and mean of squares computed together for better stability
    mean_acc = 0.0
    meansq_acc = 0.0
    for tile_idx in range(0, num_tiles):
        tile_col = tile_idx * TILE + tl.arange(0, TILE)
        offs = row * N + tile_col
        mask = tile_col < N
        x = tl.load(X + offs, mask, other=0.0)
        mean_acc += tl.sum(x, 0)
        meansq_acc += tl.sum(x * x, 0)
    mean = mean_acc / N
    meansq = meansq_acc / N
    var = meansq - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    # Second pass: normalization and write output
    for tile_idx in range(0, num_tiles):
        tile_col = tile_idx * TILE + tl.arange(0, TILE)
        offs = row * N + tile_col
        mask = tile_col < N
        x = tl.load(X + offs, mask, other=0.0)
        w = tl.load(W + tile_col, mask, other=0.0)
        b = tl.load(B + tile_col, mask, other=0.0)
        norm = (x - mean) * inv * w + b
        tl.store(Y + offs, norm, mask)
