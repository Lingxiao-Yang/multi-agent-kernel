import triton
import triton.language as tl

@triton.jit
def layernorm_fused_fullrow_blockaccum(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    num_iters = (N + BLOCK - 1) // BLOCK

    # Allocate local accumulators (BLOCK-wide) for sum, sqr_sum, count
    sum_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    sqr_acc = tl.zeros((BLOCK,), dtype=tl.float32)
    count_acc = tl.zeros((BLOCK,), dtype=tl.float32)

    # First pass: accumulate sum, sqr_sum, count (masked)
    for i in range(num_iters):
        col = offs + i * BLOCK
        mask = col < N
        x = tl.load(X + row * N + col, mask=mask, other=0.0)
        sum_acc += tl.where(mask, x, 0.0)
        sqr_acc += tl.where(mask, x * x, 0.0)
        count_acc += tl.where(mask, 1.0, 0.0)

    # Reduce to scalars for the row
    row_sum = tl.sum(sum_acc, 0)
    row_sqr = tl.sum(sqr_acc, 0)
    row_count = tl.sum(count_acc, 0)
    mean = tl.div_rn(row_sum, row_count)
    var = tl.div_rn(row_sqr, row_count) - mean * mean
    inv = tl.rsqrt(var + 1e-5)

    # Second pass: normalize/affine
    for i in range(num_iters):
        col = offs + i * BLOCK
        mask = col < N
        x = tl.load(X + row * N + col, mask=mask, other=0.0)
        centered = x - mean
        w = tl.load(W + col, mask=mask, other=0.0)
        b = tl.load(B + col, mask=mask, other=0.0)
        y = (centered * inv) * w + b
        tl.store(Y + row * N + col, y, mask=mask)
