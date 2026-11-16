import triton
import triton.language as tl

@triton.jit
def layernorm_split4_interleaved_unroll(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    SPLITS = 4
    chunk = split * BLOCK
    cols = tl.arange(0, BLOCK) + chunk
    mask = cols < N
    offs = row * N + cols
    # Interleaved reduction: each split processes BLOCK elements offset by split, unrolling for ILP
    mean_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, N, BLOCK * SPLITS):
        for u in range(SPLITS):
            icol = off + ((split + u) % SPLITS) * BLOCK + tl.arange(0, BLOCK)
            imask = icol < N
            ioffs = row * N + icol
            x = tl.load(X + ioffs, mask=imask, other=0.0)
            mean_acc += tl.where(imask, x, 0.0)
    mean = tl.sum(mean_acc, axis=0)
    mean = tl.div_rn(mean, N)
    # Variance pass
    var_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for off in range(0, N, BLOCK * SPLITS):
        for u in range(SPLITS):
            icol = off + ((split + u) % SPLITS) * BLOCK + tl.arange(0, BLOCK)
            imask = icol < N
            ioffs = row * N + icol
            x = tl.load(X + ioffs, mask=imask, other=0.0)
            xc = x - mean
            var_acc += tl.where(imask, xc * xc, 0.0)
    var = tl.sum(var_acc, axis=0)
    var = tl.div_rn(var, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # Normalize and write only this split's region
    x = tl.load(X + offs, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    xc = x - mean
    y = (xc * inv_std) * w + b
    tl.store(Y + offs, y, mask=mask)
