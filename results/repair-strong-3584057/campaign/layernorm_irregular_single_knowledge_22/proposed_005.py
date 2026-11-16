import triton
import triton.language as tl

@triton.jit
def layernorm_fused_blockloop_fusedstats(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mean = 0.0
    mean2 = 0.0
    for off in range(0, N, BLOCK):
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
    for off in range(0, N, BLOCK):
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
