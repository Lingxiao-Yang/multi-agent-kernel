import triton
import triton.language as tl

@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0)
    mean = tl.div_rn(tl.sum(tl.where(mask, x, 0), 0), N)
    centered = tl.where(mask, x - mean, 0)
    inv = tl.rsqrt(tl.sum(centered * centered, 0) / N + 1e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    row = i // N
    col = i % N
    x = tl.load(X + i, mask, other=0)
    mean = tl.load(S + 2 * row, mask, other=0)
    inv = tl.load(S + 2 * row + 1, mask, other=0)
    w = tl.load(W + col, mask, other=0)
    b = tl.load(B + col, mask, other=0)
    y = (x - mean) * inv * w + b
    tl.store(Y + i, y, mask)
