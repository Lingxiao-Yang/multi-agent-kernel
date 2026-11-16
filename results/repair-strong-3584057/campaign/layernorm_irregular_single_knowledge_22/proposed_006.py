import triton
import triton.language as tl

@triton.jit
def layernorm_fused_blockloop_unrolled(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mean = 0.0
    mean2 = 0.0
    n_full = (N // (BLOCK * 2)) * (BLOCK * 2)
    # Unroll by 2 BLOCKs for better memory throughput
    for off in range(0, n_full, BLOCK * 2):
        offs0 = off + tl.arange(0, BLOCK)
        offs1 = off + BLOCK + tl.arange(0, BLOCK)
        mask0 = offs0 < N
        mask1 = offs1 < N
        x_ptrs0 = X + row * N + offs0
        x_ptrs1 = X + row * N + offs1
        x0 = tl.load(x_ptrs0, mask=mask0, other=0)
        x1 = tl.load(x_ptrs1, mask=mask1, other=0)
        mean += tl.sum(x0, 0) + tl.sum(x1, 0)
        mean2 += tl.sum(x0 * x0, 0) + tl.sum(x1 * x1, 0)
    # Tail block(s)
    for off in range(n_full, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0)
        mean += tl.sum(x, 0)
        mean2 += tl.sum(x * x, 0)
    mean = tl.div_rn(mean, N)
    mean2 = tl.div_rn(mean2, N)
    var = mean2 - mean * mean
    inv_std = tl.rsqrt(var + 1e-5)
    # Output pass, unrolled
    for off in range(0, n_full, BLOCK * 2):
        offs0 = off + tl.arange(0, BLOCK)
        offs1 = off + BLOCK + tl.arange(0, BLOCK)
        mask0 = offs0 < N
        mask1 = offs1 < N
        x_ptrs0 = X + row * N + offs0
        x_ptrs1 = X + row * N + offs1
        x0 = tl.load(x_ptrs0, mask=mask0, other=0)
        x1 = tl.load(x_ptrs1, mask=mask1, other=0)
        w0 = tl.load(W + offs0, mask=mask0, other=0)
        w1 = tl.load(W + offs1, mask=mask1, other=0)
        b0 = tl.load(B + offs0, mask=mask0, other=0)
        b1 = tl.load(B + offs1, mask=mask1, other=0)
        centered0 = tl.where(mask0, x0 - mean, 0)
        centered1 = tl.where(mask1, x1 - mean, 0)
        y0 = (centered0 * inv_std) * w0 + b0
        y1 = (centered1 * inv_std) * w1 + b1
        y_ptrs0 = Y + row * N + offs0
        y_ptrs1 = Y + row * N + offs1
        tl.store(y_ptrs0, y0, mask0)
        tl.store(y_ptrs1, y1, mask1)
    # Tail output block(s)
    for off in range(n_full, N, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0)
        w = tl.load(W + offs, mask=mask, other=0)
        b = tl.load(B + offs, mask=mask, other=0)
        centered = tl.where(mask, x - mean, 0)
        y = (centered * inv_std) * w + b
        y_ptrs = Y + row * N + offs
        tl.store(y_ptrs, y, mask)
