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
