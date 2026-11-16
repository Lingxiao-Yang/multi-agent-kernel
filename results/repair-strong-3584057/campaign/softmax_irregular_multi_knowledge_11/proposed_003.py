import triton
import triton.language as tl

@triton.jit
def softmax_split_tile(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    TILE = BLOCK // 2
    col_offset = tile_id * TILE
    cols = col_offset + tl.arange(0, TILE)
    offs = row * N + cols
    # Load input for the tile with masking
    x_tile = tl.load(X + offs, cols < N, other=-float('inf'))
    # Local maximum in this tile
    tile_max = tl.max(x_tile, 0)
    # Shared memory for tile maxima -- use local arrays to simulate communication
    max_tiles = tl.zeros((2,), dtype=tl.float32)
    max_tiles = tl.where(tl.arange(0, 2) == tile_id, tile_max, max_tiles)
    # All threads in the row exchange their tile_max via reduction
    row_max = tl.maximum(max_tiles[0], max_tiles[1])
    # Center and exponentiate
    x_centered = x_tile - row_max
    exp_x = tl.exp(x_centered)
    # Local sum for this tile
    tile_sum = tl.sum(exp_x, 0)
    sum_tiles = tl.zeros((2,), dtype=tl.float32)
    sum_tiles = tl.where(tl.arange(0, 2) == tile_id, tile_sum, sum_tiles)
    row_sum = sum_tiles[0] + sum_tiles[1]
    # Normalize output
    y_tile = exp_x / row_sum
    tl.store(Y + offs, y_tile, cols < N)
