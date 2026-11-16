import triton
import triton.language as tl

@triton.jit
def layernorm_partial_reduction_v2(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    seg = tl.program_id(1)
    n_segs = 2
    seg_size = (N + n_segs - 1) // n_segs
    seg_start = seg * seg_size
    seg_end = tl.minimum(seg_start + seg_size, N)
    col = tl.arange(0, BLOCK)
    cols = seg_start + col
    seg_mask = (cols >= seg_start) & (cols < seg_end) & (cols < N)
    offs_X = row * N + cols
    x = tl.load(X + offs_X, seg_mask, other=0)
    valid_count = tl.sum(seg_mask, 0)
    part_sum = tl.sum(tl.where(seg_mask, x, 0), 0)
    part_sq_sum = tl.sum(tl.where(seg_mask, x * x, 0), 0)
    # Each row gets 6 slots: [sum0, sumsq0, count0, sum1, sumsq1, count1]
    offs_S = row * 6 + seg * 3
    tl.store(S + offs_S + 0, part_sum)
    tl.store(S + offs_S + 1, part_sq_sum)
    tl.store(S + offs_S + 2, valid_count)

@triton.jit
def layernorm_finalize_v2(S, W, B, X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    sum0 = tl.load(S + row * 6 + 0)
    sq_sum0 = tl.load(S + row * 6 + 1)
    cnt0 = tl.load(S + row * 6 + 2)
    sum1 = tl.load(S + row * 6 + 3)
    sq_sum1 = tl.load(S + row * 6 + 4)
    cnt1 = tl.load(S + row * 6 + 5)
    total_sum = sum0 + sum1
    total_sq_sum = sq_sum0 + sq_sum1
    total_count = cnt0 + cnt1
    N_eff = tl.where(total_count > 0, total_count, 1)
    mean = tl.div_rn(total_sum, N_eff)
    var = tl.div_rn(total_sq_sum, N_eff) - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    col = tl.arange(0, BLOCK)
    offs = row * N + col
    mask = col < N
    x = tl.load(X + offs, mask, other=0)
    w = tl.load(W + col, mask, other=0)
    b = tl.load(B + col, mask, other=0)
    norm = tl.where(mask, (x - mean) * inv * w + b, 0)
    tl.store(Y + offs, norm, mask)
