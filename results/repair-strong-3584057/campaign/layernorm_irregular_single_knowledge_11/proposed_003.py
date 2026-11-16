import triton
import triton.language as tl

@triton.jit
def layernorm_row_chunked_parallel(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    # Accumulate mean and var in chunks
    mean_acc = tl.zeros((), dtype=tl.float32)
    var_acc = tl.zeros((), dtype=tl.float32)
    # First pass: mean
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        offs = row * N + cols
        x = tl.load(X + offs, mask=mask, other=0.0)
        mean_acc += tl.sum(x, axis=0)
    mean = tl.div_rn(mean_acc, N)
    # Second pass: variance
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        offs = row * N + cols
        x = tl.load(X + offs, mask=mask, other=0.0)
        xc = x - mean
        xc = tl.where(mask, xc, 0.0)
        var_acc += tl.sum(xc * xc, axis=0)
    var = tl.div_rn(var_acc, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # Final pass: normalize and write
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        offs = row * N + cols
        x = tl.load(X + offs, mask=mask, other=0.0)
        w = tl.load(W + cols, mask=mask, other=0.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        xc = x - mean
        y = (xc * inv_std) * w + b
        tl.store(Y + offs, y, mask=mask)
