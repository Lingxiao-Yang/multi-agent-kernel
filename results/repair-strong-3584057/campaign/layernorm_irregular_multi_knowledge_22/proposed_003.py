import triton
import triton.language as tl

@triton.jit
def layernorm_row_split(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    split = tl.program_id(1)
    chunk = split * BLOCK
    offs = chunk + tl.arange(0, BLOCK)
    mask = offs < N
    x_ptr = X + row_id * N + offs
    x = tl.load(x_ptr, mask, other=0)
    part_sum = tl.sum(x, 0)
    part_sumsq = tl.sum(x * x, 0)

    # Use per-row shared memory for 2 splits (each writes two values)
    smem = tl.zeros((2, 2), dtype=tl.float32)
    # Each split writes partials to shared memory
    tl.store(smem + split * 2 + 0, part_sum)
    tl.store(smem + split * 2 + 1, part_sumsq)
    tl.barrier()
    mean = 0.0
    inv_std = 1.0
    if split == 0:
        sum0 = tl.load(smem + 0)
        sum1 = tl.load(smem + 2)
        sumsq0 = tl.load(smem + 1)
        sumsq1 = tl.load(smem + 3)
        full_sum = sum0 + sum1
        full_sumsq = sumsq0 + sumsq1
        mean = tl.div_rn(full_sum, N)
        # E[x^2] - (E[x])^2 for variance
        mean_sq = mean * mean
        var = tl.div_rn(full_sumsq, N) - mean_sq
        inv_std = tl.rsqrt(var + 1e-5)
    # Broadcast mean/inv_std to all threads in both splits
    mean = tl.broadcast_to(mean, (BLOCK,))
    inv_std = tl.broadcast_to(inv_std, (BLOCK,))
    # Reload and normalize
    x = tl.load(x_ptr, mask, other=0)
    w = tl.load(W + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    norm = (x - mean) * inv_std
    y = norm * w + b
    tl.store(Y + row_id * N + offs, y, mask)
