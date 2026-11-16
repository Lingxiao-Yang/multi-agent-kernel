import triton
import triton.language as tl

@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    sum_val = 0.0
    sumsq_val = 0.0
    for start in range(0, N, BLOCK):
        idx = start + offs
        mask = idx < N
        x = tl.load(X + row * N + idx, mask, other=0.0)
        x32 = tl.where(mask, x, 0.0)
        sum_val += tl.sum(x32, 0)
        sumsq_val += tl.sum(x32 * x32, 0)
    mean = tl.div_rn(sum_val, N)
    var = tl.div_rn(sumsq_val, N) - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    row = i // N
    col = i % N
    x = tl.load(X + i, mask, other=0.0)
    mean = tl.load(S + row * 2, mask, other=0.0)
    inv = tl.load(S + row * 2 + 1, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)
    val = (x - mean) * inv * w + b
    tl.store(Y + i, val, mask)
