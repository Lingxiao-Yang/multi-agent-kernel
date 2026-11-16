import triton
import triton.language as tl

@triton.jit
def layernorm_fused_blockloop_vec(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n_blocks = (N + BLOCK - 1) // BLOCK
    mean_acc = 0.0
    for block_idx in range(n_blocks):
        off = block_idx * BLOCK
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0)
        mean_acc += tl.sum(x, 0)
    mean = tl.div_rn(mean_acc, N)
    var_acc = 0.0
    for block_idx in range(n_blocks):
        off = block_idx * BLOCK
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0)
        centered = tl.where(mask, x - mean, 0)
        var_acc += tl.sum(centered * centered, 0)
    var = var_acc / N
    inv_std = tl.rsqrt(var + 1e-5)
    # Vectorized output loop
    for block_idx in range(n_blocks):
        off = block_idx * BLOCK
        offs = off + tl.arange(0, BLOCK)
        mask = offs < N
        x_ptrs = X + row * N + offs
        x = tl.load(x_ptrs, mask=mask, other=0)
        centered = tl.where(mask, x - mean, 0)
        w = tl.load(W + offs, mask=mask, other=0)
        b = tl.load(B + offs, mask=mask, other=0)
        y = (centered * inv_std) * w + b
        y_ptrs = Y + row * N + offs
        tl.store(y_ptrs, y, mask)
