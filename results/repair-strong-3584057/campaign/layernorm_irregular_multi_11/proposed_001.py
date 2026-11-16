import triton
import triton.language as tl

@triton.jit
def layernorm_stats_split(X, S, N: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask, other=0)
    sum_x = tl.sum(x, 0)
    sum_x2 = tl.sum(x * x, 0)
    # Write partial sums to scratch S
    tl.store(S + row * 2 * SPLIT + split * 2 + 0, sum_x)
    tl.store(S + row * 2 * SPLIT + split * 2 + 1, sum_x2)
    # Let split==0 do the global row reduction
    if split == 0:
        total_sum = tl.zeros((), dtype=tl.float32)
        total_sum2 = tl.zeros((), dtype=tl.float32)
        for k in range(SPLIT):
            total_sum += tl.load(S + row * 2 * SPLIT + k * 2 + 0)
            total_sum2 += tl.load(S + row * 2 * SPLIT + k * 2 + 1)
        mean = tl.div_rn(total_sum, N)
        var = tl.div_rn(total_sum2, N) - mean * mean
        inv = tl.rsqrt(var + 1e-5)
        tl.store(S + 2 * row, mean)
        tl.store(S + 2 * row + 1, inv)

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
