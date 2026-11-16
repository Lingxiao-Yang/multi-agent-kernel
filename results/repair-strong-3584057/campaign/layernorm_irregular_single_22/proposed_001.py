import triton
import triton.language as tl

@triton.jit
def layernorm_blockwise(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # Blockwise mean and variance reduction
    mean = tl.sum(x, 0) / N
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, 0) / N
    inv = tl.rsqrt(var + 1e-5)
    # Affine transform
    w = tl.load(W + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = (centered * inv) * w + b
    tl.store(Y + row * N + offs, y, mask=mask)
    # Loop to cover all columns if N > BLOCK
    num_iters = (N + BLOCK - 1) // BLOCK
    for iter in range(1, num_iters):
        offs_iter = offs + iter * BLOCK
        mask_iter = offs_iter < N
        x_ptrs_iter = X + row * N + offs_iter
        x_iter = tl.load(x_ptrs_iter, mask=mask_iter, other=0.0)
        centered_iter = tl.where(mask_iter, x_iter - mean, 0.0)
        w_iter = tl.load(W + offs_iter, mask=mask_iter, other=0.0)
        b_iter = tl.load(B + offs_iter, mask=mask_iter, other=0.0)
        y_iter = (centered_iter * inv) * w_iter + b_iter
        tl.store(Y + row * N + offs_iter, y_iter, mask=mask_iter)
