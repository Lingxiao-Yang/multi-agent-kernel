import triton
import triton.language as tl

@triton.jit
def layernorm_fused_blockloop_unrolled4(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mean = 0.0
    mean2 = 0.0
    unroll = 4
    n_full = (N // (BLOCK * unroll)) * (BLOCK * unroll)
    # Unrolled reduction by 4*BLOCK
    for off in range(0, n_full, BLOCK * unroll):
        offs0 = off + tl.arange(0, BLOCK)
        offs1 = off + BLOCK + tl.arange(0, BLOCK)
        offs2 = off + 2*BLOCK + tl.arange(0, BLOCK)
        offs3 = off + 3*BLOCK + tl.arange(0, BLOCK)
        mask0 = offs0 < N
        mask1 = offs1 < N
        mask2 = offs2 < N
        mask3 = offs3 < N
        x_ptrs0 = X + row * N + offs0
        x_ptrs1 = X + row * N + offs1
        x_ptrs2 = X + row * N + offs2
        x_ptrs3 = X + row * N + offs3
        x0 = tl.load(x_ptrs0, mask=mask0, other=0)
        x1 = tl.load(x_ptrs1, mask=mask1, other=0)
        x2 = tl.load(x_ptrs2, mask=mask2, other=0)
        x3 = tl.load(x_ptrs3, mask=mask3, other=0)
        mean += tl.sum(x0, 0) + tl.sum(x1, 0) + tl.sum(x2, 0) + tl.sum(x3, 0)
        mean2 += tl.sum(x0 * x0, 0) + tl.sum(x1 * x1, 0) + tl.sum(x2 * x2, 0) + tl.sum(x3 * x3, 0)
    # Tail blocks
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
    # Unrolled output by 4*BLOCK
    for off in range(0, n_full, BLOCK * unroll):
        offs0 = off + tl.arange(0, BLOCK)
        offs1 = off + BLOCK + tl.arange(0, BLOCK)
        offs2 = off + 2*BLOCK + tl.arange(0, BLOCK)
        offs3 = off + 3*BLOCK + tl.arange(0, BLOCK)
        mask0 = offs0 < N
        mask1 = offs1 < N
        mask2 = offs2 < N
        mask3 = offs3 < N
        x_ptrs0 = X + row * N + offs0
        x_ptrs1 = X + row * N + offs1
        x_ptrs2 = X + row * N + offs2
        x_ptrs3 = X + row * N + offs3
        x0 = tl.load(x_ptrs0, mask=mask0, other=0)
        x1 = tl.load(x_ptrs1, mask=mask1, other=0)
        x2 = tl.load(x_ptrs2, mask=mask2, other=0)
        x3 = tl.load(x_ptrs3, mask=mask3, other=0)
        w0 = tl.load(W + offs0, mask=mask0, other=0)
        w1 = tl.load(W + offs1, mask=mask1, other=0)
        w2 = tl.load(W + offs2, mask=mask2, other=0)
        w3 = tl.load(W + offs3, mask=mask3, other=0)
        b0 = tl.load(B + offs0, mask=mask0, other=0)
        b1 = tl.load(B + offs1, mask=mask1, other=0)
        b2 = tl.load(B + offs2, mask=mask2, other=0)
        b3 = tl.load(B + offs3, mask=mask3, other=0)
        centered0 = tl.where(mask0, x0 - mean, 0)
        centered1 = tl.where(mask1, x1 - mean, 0)
        centered2 = tl.where(mask2, x2 - mean, 0)
        centered3 = tl.where(mask3, x3 - mean, 0)
        y0 = (centered0 * inv_std) * w0 + b0
        y1 = (centered1 * inv_std) * w1 + b1
        y2 = (centered2 * inv_std) * w2 + b2
        y3 = (centered3 * inv_std) * w3 + b3
        y_ptrs0 = Y + row * N + offs0
        y_ptrs1 = Y + row * N + offs1
        y_ptrs2 = Y + row * N + offs2
        y_ptrs3 = Y + row * N + offs3
        tl.store(y_ptrs0, y0, mask0)
        tl.store(y_ptrs1, y1, mask1)
        tl.store(y_ptrs2, y2, mask2)
        tl.store(y_ptrs3, y3, mask3)
    # Tail output blocks
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
