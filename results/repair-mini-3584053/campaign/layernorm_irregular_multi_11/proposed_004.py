import triton
import triton.language as tl

@triton.jit
def layernorm_stats_part1(X, S, N: tl.constexpr, BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    row = tl.program_id(axis=0)
    block_id = tl.program_id(axis=1)
    offs = tl.arange(0, BLOCK)
    col = block_id * BLOCK + offs
    mask = col < N
    ptr = row * N + col
    x = tl.load(X + ptr, mask, other=0.0)
    partial_sum = tl.sum(x, 0)
    idx = row * NUM_BLOCKS + block_id
    tl.store(S + idx, partial_sum, True)

@triton.jit
def layernorm_stats_part2(X, S, N: tl.constexpr, BLOCK: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    row = tl.program_id(axis=0)
    offs = tl.arange(0, NUM_BLOCKS)
    mask = offs < NUM_BLOCKS
    ptr = row * NUM_BLOCKS + offs
    partial_sums = tl.load(S + ptr, mask, other=0.0)
    total_sum = tl.sum(partial_sums, 0)
    mean = tl.div_rn(total_sum, N)

    sum_sq = tl.float32(0.0)
    for i in range(NUM_BLOCKS):
        col = i * BLOCK + tl.arange(0, BLOCK)
        mask_x = col < N
        ptr_x = row * N + col
        x = tl.load(X + ptr_x, mask_x, other=0.0)
        centered = tl.where(mask_x, x - mean, 0.0)
        sum_sq += tl.sum(centered * centered, 0)

    inv = tl.rsqrt(tl.div_rn(sum_sq, N) + 1e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    row = i // N
    col = i % N
    x = tl.load(X + i, mask, other=0.0)
    mean = tl.load(S + 2 * row, mask, other=0.0)
    inv = tl.load(S + 2 * row + 1, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)
    val = (x - mean) * inv * w + b
    tl.store(Y + i, val, mask)

# Grid sizes and kernel arguments are concretely set for the problem shape (M=513, N=781) with BLOCK=512
# NUM_BLOCKS = ceil(781/512) = 2
# layernorm_stats_part1 grid: [513 (rows), 2 (blocks per row)]
# layernorm_stats_part2 grid: [513 (rows)]
# layernorm_apply grid: ceil((513*781)/512) = 784
# This implements a valid hierarchical reduction without unsupported primitives and uses masking
# for both partial sums and final sums, and an increased BLOCK size of 512 for better occupancy
