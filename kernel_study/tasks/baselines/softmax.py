import triton
import triton.language as tl

@triton.jit
def row_partials(X, S, ROWS: tl.constexpr, N: tl.constexpr, GROUPS: tl.constexpr,
                 BLOCK: tl.constexpr, SOFTMAX: tl.constexpr):
    group = tl.program_id(0)
    row = tl.program_id(1)
    columns = group * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + row*N + columns, columns < N, other=0)
    if SOFTMAX:
        first = tl.max(tl.where(columns < N, x, -float('inf')), 0)
        second = tl.sum(tl.where(columns < N, tl.exp(x-first), 0), 0)
    else:
        first = tl.sum(x, 0) / tl.minimum(BLOCK, N-group*BLOCK)
        centered = tl.where(columns < N, x-first, 0)
        second = tl.sum(centered * centered, 0)
    slot = 2 * (ROWS + row*GROUPS + group)
    tl.store(S + slot, first)
    tl.store(S + slot + 1, second)

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
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    x = tl.load(X + i, i < TOTAL, other=0)
    mx = tl.load(S + row * 2, i < TOTAL, other=0)
    denom = tl.load(S + row * 2 + 1, i < TOTAL, other=1)
    tl.store(Y + i, tl.exp(x - mx) / denom, i < TOTAL)

import torch
def kernel_function(X):
    Y = X.new_empty((4096, 393216))
    S = Y.new_empty((3153920,), dtype=torch.float32)
    row_partials[(384, 4096)](X, S, 4096, 393216, 384, 1024, 1, num_warps=4, num_stages=1)
    row_combine[(4096,)](S, 4096, 393216, 384, 1024, 512, 1, num_warps=4, num_stages=1)
    softmax_apply[(786432,)](X, S, Y, 1610612736, 393216, 2048, num_warps=4, num_stages=1)
    return Y
