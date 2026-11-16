import triton
import triton.language as tl

@triton.jit
def layernorm_fused_vec8(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    VEC = 8
    vec_stride = BLOCK * VEC
    mean = 0.0
    mean2 = 0.0
    n_full = (N // vec_stride) * vec_stride
    # Vectorized reduction
    for off in range(0, n_full, vec_stride):
        for v in range(VEC):
            offs = off + v * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x_ptrs = X + row * N + offs
            x = tl.load(x_ptrs, mask=mask, other=0)
            mean += tl.sum(x, 0)
            mean2 += tl.sum(x * x, 0)
    # Tail reduction
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
    # Vectorized output
    for off in range(0, n_full, vec_stride):
        for v in range(VEC):
            offs = off + v * BLOCK + tl.arange(0, BLOCK)
            mask = offs < N
            x_ptrs = X + row * N + offs
            x = tl.load(x_ptrs, mask=mask, other=0)
            w = tl.load(W + offs, mask=mask, other=0)
            b = tl.load(B + offs, mask=mask, other=0)
            centered = tl.where(mask, x - mean, 0)
            y = (centered * inv_std) * w + b
            y_ptrs = Y + row * N + offs
            tl.store(y_ptrs, y, mask)
    # Tail output
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
