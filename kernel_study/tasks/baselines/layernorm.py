import triton
import triton.language as tl

@triton.jit
def layernorm_partials(X, S, ROWS: tl.constexpr, N: tl.constexpr, GROUPS: tl.constexpr,
                       BLOCK: tl.constexpr):
    group = tl.program_id(0)
    row = tl.program_id(1)
    columns = group * BLOCK + tl.arange(0, BLOCK)
    origin = tl.load(X + row*N)
    x = tl.load(X + row*N + columns, columns < N, other=0)
    shifted = tl.where(columns < N, x-origin, 0)
    mean = tl.sum(shifted, 0) / tl.minimum(BLOCK, N-group*BLOCK)
    centered = tl.where(columns < N, shifted-mean, 0)
    slot = 2 * (ROWS + row*GROUPS + group)
    tl.store(S + slot, mean)
    tl.store(S + slot + 1, tl.sum(centered*centered, 0))

@triton.jit
def row_combine(S, ROWS: tl.constexpr, N: tl.constexpr, GROUPS: tl.constexpr,
                BLOCK: tl.constexpr, WIDTH: tl.constexpr, SOFTMAX: tl.constexpr):
    row = tl.program_id(0)
    group = tl.arange(0, WIDTH)
    slot = 2 * (ROWS + row*GROUPS + group)
    first = tl.load(S + slot, group < GROUPS, other=0)
    second = tl.load(S + slot + 1, group < GROUPS, other=0)
    if SOFTMAX:
        center = tl.max(tl.where(group < GROUPS, first, -float('inf')), 0)
        result = tl.sum(tl.where(group < GROUPS, second * tl.exp(first-center), 0), 0)
    else:
        count = tl.maximum(0, tl.minimum(BLOCK, N-group*BLOCK))
        center = tl.sum(first * count, 0)/N
        delta = first - center
        result = tl.rsqrt(tl.sum(second + count * delta * delta, 0)/N + 1e-5)
    tl.store(S + 2*row, center)
    tl.store(S + 2*row + 1, result)

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
    origin = tl.load(X + row*N, i < TOTAL, other=0)
    tl.store(Y + i, ((x - origin) - mean) * inv * w + b, i < TOTAL)

import torch
def kernel_function(X, weight, bias):
    Y = X.new_empty((16, 64, 256, 256))
    S = Y.new_empty((131104,), dtype=torch.float32)
    layernorm_partials[(4096, 16)](X, S, 16, 4194304, 4096, 1024, num_warps=4, num_stages=1)
    row_combine[(16,)](S, 16, 4194304, 4096, 1024, 4096, 0, num_warps=4, num_stages=1)
    layernorm_apply[(32768,)](X, weight, bias, S, Y, 67108864, 4194304, 2048, num_warps=4, num_stages=1)
    return Y
