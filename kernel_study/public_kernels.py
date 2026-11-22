"""Starting device functions for large public reduction and activation tasks."""
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
def gelu(X, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    x = tl.load(X+i,i<TOTAL,other=0.)
    tl.store(Y+i,0.5*x*(1.+tl.erf(x*0.7071067811865476)),i<TOTAL)


@triton.jit
def rms_norm(X, Y, C: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(1)
    feature = tl.arange(0,C)
    position = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    offset = (batch*C + feature[:,None])*SPATIAL + position[None,:]
    x = tl.load(X+offset,position[None,:]<SPATIAL,other=0.)
    scale = tl.rsqrt(tl.sum(x*x,0)/C + 1e-5)
    tl.store(Y+offset,x*scale[None,:],position[None,:]<SPATIAL)


@triton.jit
def matmul_tensor(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(K, BK)):
        ks = start * BK + kk
        a = tl.load(A + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B + ks[:, None] * N + cols[None, :],
                    (ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    x = tl.load(X + i, i < TOTAL, other=0)
    mx = tl.load(S + row * 2, i < TOTAL, other=0)
    denom = tl.load(S + row * 2 + 1, i < TOTAL, other=1)
    tl.store(Y + i, tl.exp(x - mx) / denom, i < TOTAL)


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
