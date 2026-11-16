import triton
import triton.language as tl

@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # constexpr extent
    mask = offs < N
    x = tl.load(X + row * N + offs, mask, other=0.0)
    sum_x = tl.sum(tl.where(mask, x, 0.0), 0)
    mean = tl.div_rn(sum_x, N)
    centered = tl.where(mask, x - mean, 0.0)
    sum_sq = tl.sum(centered * centered, 0)
    inv = tl.rsqrt(tl.div_rn(sum_sq, N) + 1e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply_fused_hierarchical_fixed(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    sub_block = 32  # constexpr
    num_sub_blocks = BLOCK // sub_block  # constexpr
    offs = tl.arange(0, sub_block)  # constexpr extent
    base = pid * BLOCK

    for sub_idx in range(num_sub_blocks):
        offset = base + sub_idx * sub_block + offs
        mask = offset < TOTAL
        row = offset // N
        col = offset % N

        x = tl.load(X + offset, mask, other=0.0)
        mean = tl.load(S + 2 * row, mask, other=0.0)
        inv = tl.load(S + 2 * row + 1, mask, other=0.0)
        w = tl.load(W + col, mask, other=0.0)
        b = tl.load(B + col, mask, other=0.0)

        y = (x - mean) * inv * w + b
        tl.store(Y + offset, y, mask)
