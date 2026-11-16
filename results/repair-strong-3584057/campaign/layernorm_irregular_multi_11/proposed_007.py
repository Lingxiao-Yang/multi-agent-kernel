import triton
import triton.language as tl

# First stage: Each program processes half a row, outputs partial sums to T
@triton.jit
def layernorm_stats_split2_stage1(X, T, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    row = pid // SPLIT
    split = pid % SPLIT
    offs = split * (N // SPLIT) + tl.arange(0, BLOCK)
    mask = (row < 513) & (offs < N)
    x = tl.load(X + row * N + offs, mask, other=0)
    sum_x = tl.sum(x, 0)
    sum_x2 = tl.sum(x * x, 0)
    # Write to T as [513 * SPLIT, 2] buffer
    base = (row * SPLIT + split) * 2
    tl.store(T + base, sum_x, row < 513)
    tl.store(T + base + 1, sum_x2, row < 513)

# Second stage: Reduce the partials for each row, compute mean/inv
@triton.jit
def layernorm_stats_split2_stage2(T, S, N: tl.constexpr, SPLIT: tl.constexpr):
    row = tl.program_id(0)
    sum_x = tl.zeros((), dtype=tl.float32)
    sum_x2 = tl.zeros((), dtype=tl.float32)
    for split in range(0, SPLIT):
        base = (row * SPLIT + split) * 2
        sum_x += tl.load(T + base, row < 513, other=0)
        sum_x2 += tl.load(T + base + 1, row < 513, other=0)
    mean = tl.div_rn(sum_x, N)
    var = tl.div_rn(sum_x2, N) - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    tl.store(S + 2 * row, mean, row < 513)
    tl.store(S + 2 * row + 1, inv, row < 513)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    col = i % N
    x = tl.load(X + i, i < TOTAL, other=0)
    mean = tl.load(S + row * 2, i < TOTAL, other=0)
    inv = tl.load(S + row * 2 + 1, i < TOTAL, other=0)
    w = tl.load(W + col, i < TOTAL, other=0)
    b = tl.load(B + col, i < TOTAL, other=0)
    tl.store(Y + i, (x - mean) * inv * w + b, i < TOTAL)
