import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = row * N + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < N

    x = tl.load(X + offs, mask, other=0.0)

    # Compute mean with masked sum and accurate division
    sum_x = tl.sum(tl.where(mask, x, 0.0), 0)
    mean = tl.div_rn(sum_x, N)

    # Centered x with masking
    centered = tl.where(mask, x - mean, 0.0)

    # Reapply mask on centered values before variance computation
    centered = tl.where(mask, centered, 0.0)

    sum_sq = tl.sum(centered * centered, 0)
    invvar = tl.rsqrt(tl.div_rn(sum_sq, N) + 1e-5)

    offs_col = tl.arange(0, BLOCK)
    w = tl.load(W + offs_col, mask, other=0.0)
    b = tl.load(B + offs_col, mask, other=0.0)

    y = (centered * invvar) * w + b

    tl.store(Y + offs, y, mask)
