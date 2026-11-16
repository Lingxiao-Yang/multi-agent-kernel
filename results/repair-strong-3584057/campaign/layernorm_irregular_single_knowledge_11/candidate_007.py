import triton
import triton.language as tl

@triton.jit
def layernorm_row_split4_vector(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    SPLITS = 4
    chunk = split * BLOCK
    cols = tl.arange(0, BLOCK) + chunk
    mask = cols < N
    offs = row * N + cols
    # 1. Compute mean collaboratively: all splits cover the full row by chunks of BLOCK*4
    mean_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, N, BLOCK * SPLITS):
        for s in range(SPLITS):
            bcols = off + s * BLOCK + tl.arange(0, BLOCK)
            bmask = bcols < N
            boffs = row * N + bcols
            x = tl.load(X + boffs, mask=bmask, other=0.0)
            mean_acc += tl.where(bmask, x, 0.0)
    mean = tl.sum(mean_acc, axis=0)
    mean = tl.div_rn(mean, N)
    # 2. Compute variance collaboratively
    var_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, N, BLOCK * SPLITS):
        for s in range(SPLITS):
            bcols = off + s * BLOCK + tl.arange(0, BLOCK)
            bmask = bcols < N
            boffs = row * N + bcols
            x = tl.load(X + boffs, mask=bmask, other=0.0)
            xc = x - mean
            var_acc += tl.where(bmask, xc * xc, 0.0)
    var = tl.sum(var_acc, axis=0)
    var = tl.div_rn(var, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # 3. Normalize and write only this split's region
    x = tl.load(X + offs, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    xc = x - mean
    y = (xc * inv_std) * w + b
    tl.store(Y + offs, y, mask=mask)
