import triton
import triton.language as tl

@triton.jit
def layernorm_stats_parallel(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mean_acc = tl.zeros((), dtype=tl.float32)
    m2_acc = tl.zeros((), dtype=tl.float32)
    for offs_base in range(0, N, BLOCK):
        col = offs_base + offs
        mask = col < N
        x = tl.load(X + row * N + col, mask, other=0)
        chunk_sum = tl.sum(x, 0)
        chunk_sum2 = tl.sum(x * x, 0)
        mean_acc += chunk_sum
        m2_acc += chunk_sum2
    mean = tl.div_rn(mean_acc, N)
    var = tl.div_rn(m2_acc, N) - mean * mean
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
