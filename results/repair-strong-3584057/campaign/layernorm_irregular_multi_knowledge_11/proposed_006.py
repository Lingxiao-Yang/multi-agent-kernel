import triton
import triton.language as tl

@triton.jit
def layernorm_rowwise_tiled(X, W, B, Y, N: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    num_tiles = (N + TILE - 1) // TILE
    mean_acc = 0.0
    # First pass: mean
    for tile_idx in range(0, num_tiles):
        tile_col = tile_idx * TILE + tl.arange(0, TILE)
        offs = row * N + tile_col
        mask = tile_col < N
        x = tl.load(X + offs, mask, other=0.0)
        mean_acc += tl.sum(x, 0)
    mean = mean_acc / N
    # Second pass: variance
    var_acc = 0.0
    for tile_idx in range(0, num_tiles):
        tile_col = tile_idx * TILE + tl.arange(0, TILE)
        offs = row * N + tile_col
        mask = tile_col < N
        x = tl.load(X + offs, mask, other=0.0)
        diff = x - mean
        var_acc += tl.sum(diff * diff, 0)
    var = var_acc / N
    inv = tl.rsqrt(var + 1e-5)
    # Third pass: normalization and write output
    for tile_idx in range(0, num_tiles):
        tile_col = tile_idx * TILE + tl.arange(0, TILE)
        offs = row * N + tile_col
        mask = tile_col < N
        x = tl.load(X + offs, mask, other=0.0)
        w = tl.load(W + tile_col, mask, other=0.0)
        b = tl.load(B + tile_col, mask, other=0.0)
        norm = (x - mean) * inv * w + b
        tl.store(Y + offs, norm, mask)
