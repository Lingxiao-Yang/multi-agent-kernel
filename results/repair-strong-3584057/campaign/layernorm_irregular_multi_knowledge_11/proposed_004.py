import triton
import triton.language as tl

@triton.jit
def layernorm_partial_reduction(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    seg = tl.program_id(1)
    n_segs = 2
    seg_size = (N + n_segs - 1) // n_segs
    seg_start = seg * seg_size
    seg_end = tl.minimum(seg_start + seg_size, N)
    col = tl.arange(0, BLOCK)
    cols = seg_start + col
    seg_mask = cols < seg_end
    offs_X = row * N + cols
    x = tl.load(X + offs_X, seg_mask, other=0)
    part_sum = tl.sum(tl.where(seg_mask, x, 0), 0)
    part_sq_sum = tl.sum(tl.where(seg_mask, x * x, 0), 0)
    # Each row gets 4 slots: [sum0, sumsq0, sum1, sumsq1]
    offs_S = row * 4 + seg * 2
    tl.store(S + offs_S + 0, part_sum)
    tl.store(S + offs_S + 1, part_sq_sum)

@triton.jit
def layernorm_finalize(S, W, B, X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sum0 = tl.load(S + row * 4 + 0)
    sq_sum0 = tl.load(S + row * 4 + 1)
    sum1 = tl.load(S + row * 4 + 2)
    sq_sum1 = tl.load(S + row * 4 + 3)
    total_sum = sum0 + sum1
    total_sq_sum = sq_sum0 + sq_sum1
    mean = tl.div_rn(total_sum, N)
    var = tl.div_rn(total_sq_sum, N) - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    mask = col < N
    x = tl.load(X + offs, mask, other=0)
    w = tl.load(W + col, mask, other=0)
    b = tl.load(B + col, mask, other=0)
    norm = tl.where(mask, (x - mean) * inv * w + b, 0)
    tl.store(Y + offs, norm, mask)
