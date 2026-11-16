import triton
import triton.language as tl

@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mean_acc = 0.0
    var_acc = 0.0
    count = 0.0

    # Loop over row in chunks of BLOCK to cover all columns
    for block_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = block_start + offs
        mask = col < N
        x = tl.load(X + row * N + col, mask, other=0.0)
        mean_acc += tl.sum(tl.where(mask, x, 0.0), 0)
        count += tl.sum(tl.where(mask, 1.0, 0.0), 0)

    mean = mean_acc / count

    # Accumulate variance across the full row
    for block_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = block_start + offs
        mask = col < N
        x = tl.load(X + row * N + col, mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        var_acc += tl.sum(centered * centered, 0)

    var = var_acc / count
    inv_std = tl.rsqrt(var + 1e-5)

    # Normalize and write output in chunks
    for block_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = block_start + offs
        mask = col < N
        x = tl.load(X + row * N + col, mask, other=0.0)
        w = tl.load(W + col, mask, other=0.0)
        b = tl.load(B + col, mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(Y + row * N + col, y, mask)
