import triton
import triton.language as tl

@triton.jit
def layernorm_row_multiwarp(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    # Per-lane accumulators for mean and variance
    mean_acc = tl.zeros((), dtype=tl.float32)
    var_acc = tl.zeros((), dtype=tl.float32)
    # Strided reduction for mean
    for c in range(lane, N, BLOCK):
        mask = c < N
        offs = row * N + c
        x = tl.load(X + offs, mask=mask, other=0.0)
        mean_acc += x
    mean = tl.sum(mean_acc, axis=0)
    mean = tl.div_rn(mean, N)
    # Strided reduction for variance
    for c in range(lane, N, BLOCK):
        mask = c < N
        offs = row * N + c
        x = tl.load(X + offs, mask=mask, other=0.0)
        xc = x - mean
        xc = tl.where(mask, xc, 0.0)
        var_acc += xc * xc
    var = tl.sum(var_acc, axis=0)
    var = tl.div_rn(var, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # Final pass: normalize and write
    for c in range(lane, N, BLOCK):
        mask = c < N
        offs = row * N + c
        x = tl.load(X + offs, mask=mask, other=0.0)
        w = tl.load(W + c, mask=mask, other=0.0)
        b = tl.load(B + c, mask=mask, other=0.0)
        xc = x - mean
        y = (xc * inv_std) * w + b
        tl.store(Y + offs, y, mask=mask)
