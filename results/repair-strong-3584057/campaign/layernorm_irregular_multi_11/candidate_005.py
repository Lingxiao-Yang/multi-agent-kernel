import triton
import triton.language as tl

@triton.jit
def layernorm_stats_split2(X, S, N: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    chunk = (N + SPLIT - 1) // SPLIT
    start = split * chunk
    end = tl.minimum(start + chunk, N)
    offs = start + tl.arange(0, BLOCK)
    mask = offs < end
    x = tl.load(X + row * N + offs, mask, other=0)
    partial_sum = tl.sum(x, 0)
    partial_sum2 = tl.sum(x * x, 0)
    # Write partials to S as workspace, shape [ROWS,2*SPLIT], but only final output is [ROWS,2]
    tl.store(S + row * 2 * SPLIT + 2 * split, partial_sum)
    tl.store(S + row * 2 * SPLIT + 2 * split + 1, partial_sum2)
    # Only split==0 does the cross-split reduction
    if split == 0:
        total_sum = tl.zeros((), dtype=tl.float32)
        total_sum2 = tl.zeros((), dtype=tl.float32)
        for s in range(SPLIT):
            sx = tl.load(S + row * 2 * SPLIT + 2 * s)
            sx2 = tl.load(S + row * 2 * SPLIT + 2 * s + 1)
            total_sum += sx
            total_sum2 += sx2
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
