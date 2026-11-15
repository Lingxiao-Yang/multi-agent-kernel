import triton
import triton.language as tl

@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0.0)
    sum_x = tl.sum(x, 0)
    mean = sum_x / N
    centered = tl.where(mask, x - mean, 0.0)
    sum_sq = tl.sum(centered * centered, 0)
    inv = tl.rsqrt(sum_sq / N + 1e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    i = pid * BLOCK + offs
    mask = i < TOTAL
    row = i // N
    col = i % N
    x = tl.load(X + i, mask, other=0.0)
    mean = tl.load(S + 2 * row, mask, other=0.0)
    inv = tl.load(S + 2 * row + 1, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)
    normalized = (x - mean) * inv
    y = normalized * w + b
    tl.store(Y + i, y, mask)
