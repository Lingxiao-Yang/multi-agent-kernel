import triton
import triton.language as tl

@triton.jit
def layernorm_stats_part1(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Compute partial sums of X[row, :] in chunks of BLOCK
    num_blocks = (N + BLOCK - 1) // BLOCK
    block_id = tl.program_id(1) if tl.num_programs() > 1 else 0
    offs = tl.arange(0, BLOCK)
    col = block_id * BLOCK + offs
    mask = col < N
    ptr = row * N + col
    x = tl.load(X + ptr, mask, other=0.0)
    partial_sum = tl.sum(x, 0)
    # Store partial sums in S[row, block_id]
    tl.store(S + row * num_blocks + block_id, partial_sum, True)

@triton.jit
def layernorm_stats_part2(X, S, N: tl.constexpr):
    row = tl.program_id(0)
    # Number of partial sums per row
    BLOCK = 256  # partial sums block size used in part1
    num_blocks = (N + BLOCK - 1) // BLOCK
    offs = tl.arange(0, num_blocks)
    mask = offs < num_blocks
    ptr = row * num_blocks + offs
    partial_sums = tl.load(S + ptr, mask, other=0.0)
    total_sum = tl.sum(partial_sums, 0)
    mean = total_sum / N
    # Load full row to compute variance sum
    col = tl.arange(0, N)
    mask_x = col < N
    x = tl.load(X + row * N + col, mask_x, other=0.0)
    centered = x - mean
    sum_sq = tl.sum(centered * centered, 0)
    inv = tl.rsqrt(sum_sq / N + 1e-5)
    # Store final mean and inv per row
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

# Launch layernorm_stats_part1 with grid=[513, num_blocks], num_warps=4, num_stages=1
# Launch layernorm_stats_part2 with grid=[513], num_warps=4, num_stages=1
# Launch layernorm_apply with grid=[(513*781 + 512 - 1)//512 = 784], num_warps=4, num_stages=1
# This implements a valid hierarchical reduction via two kernel launches without unsupported primitives
# and applies layernorm with increased BLOCK=512 for better occupancy.
