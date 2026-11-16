import triton
import triton.language as tl

@triton.jit
def layernorm_row_strided_vector(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mean_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    var_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    # Strided vector reduction for mean
    for off in range(0, N, BLOCK):
        idx = off + cols
        mask = idx < N
        offs = row * N + idx
        x = tl.load(X + offs, mask=mask, other=0.0)
        mean_acc += tl.where(mask, x, 0.0)
    mean = tl.sum(mean_acc, axis=0)
    mean = tl.div_rn(mean, N)
    # Strided vector reduction for variance
    for off in range(0, N, BLOCK):
        idx = off + cols
        mask = idx < N
        offs = row * N + idx
        x = tl.load(X + offs, mask=mask, other=0.0)
        xc = x - mean
        var_acc += tl.where(mask, xc * xc, 0.0)
    var = tl.sum(var_acc, axis=0)
    var = tl.div_rn(var, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # Final strided vector pass: normalize and write
    for off in range(0, N, BLOCK):
        idx = off + cols
        mask = idx < N
        offs = row * N + idx
        x = tl.load(X + offs, mask=mask, other=0.0)
        w = tl.load(W + idx, mask=mask, other=0.0)
        b = tl.load(B + idx, mask=mask, other=0.0)
        xc = x - mean
        y = (xc * inv_std) * w + b
        tl.store(Y + offs, y, mask=mask)
