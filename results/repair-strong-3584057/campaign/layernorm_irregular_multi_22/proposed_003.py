import triton
import triton.language as tl

@triton.jit
def fused_layernorm_rowwise_2d(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    split = tl.program_id(1)
    offs = split * BLOCK
    cols = tl.arange(0, BLOCK) + offs
    mask = cols < N
    x_ptrs = X + row * N + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    # Partial sum and sumsq for this split
    part_sum = tl.sum(x, 0)
    part_sumsq = tl.sum(x * x, 0)
    # Use shared memory for inter-split reduction and broadcast, size 2 per row
    sum_smem = tl.zeros((2,), dtype=tl.float32)
    sumsq_smem = tl.zeros((2,), dtype=tl.float32)
    # Store partials
    tl.store(sum_smem + split, part_sum)
    tl.store(sumsq_smem + split, part_sumsq)
    tl.barrier()
    # Only one split computes mean/var
    if split == 0:
        total_sum = tl.load(sum_smem + 0) + tl.load(sum_smem + 1)
        total_sumsq = tl.load(sumsq_smem + 0) + tl.load(sumsq_smem + 1)
        mean = tl.div_rn(total_sum, N)
        var = tl.div_rn(total_sumsq, N) - mean * mean
        inv_std = tl.rsqrt(var + 1e-5)
        tl.store(sum_smem + 0, mean)
        tl.store(sum_smem + 1, inv_std)
    tl.barrier()
    mean = tl.load(sum_smem + 0)
    inv_std = tl.load(sum_smem + 1)
    centered = tl.where(mask, x - mean, 0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    normed = (centered * inv_std) * w + b
    tl.store(Y + row * N + cols, normed, mask=mask)
