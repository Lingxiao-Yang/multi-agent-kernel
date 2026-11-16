import triton
import triton.language as tl

@triton.jit
def layernorm_fused_fullrow_blockaccum_tail_unroll2(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    num_iters = (N // (2 * BLOCK))
    tail_start = num_iters * 2 * BLOCK
    # Accumulators for main unrolled loop
    sum_acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    sum_acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    sqr_acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    sqr_acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    count_acc0 = tl.zeros((BLOCK,), dtype=tl.float32)
    count_acc1 = tl.zeros((BLOCK,), dtype=tl.float32)
    # Main unrolled accumulation
    for i in range(num_iters):
        base = 2 * i * BLOCK
        col0 = offs + base
        col1 = offs + base + BLOCK
        mask0 = col0 < N
        mask1 = col1 < N
        x0 = tl.load(X + row * N + col0, mask=mask0, other=0.0)
        x1 = tl.load(X + row * N + col1, mask=mask1, other=0.0)
        sum_acc0 += tl.where(mask0, x0, 0.0)
        sum_acc1 += tl.where(mask1, x1, 0.0)
        sqr_acc0 += tl.where(mask0, x0 * x0, 0.0)
        sqr_acc1 += tl.where(mask1, x1 * x1, 0.0)
        count_acc0 += tl.where(mask0, 1.0, 0.0)
        count_acc1 += tl.where(mask1, 1.0, 0.0)
    # Tail processing for remaining elements
    tail_sum = tl.zeros((BLOCK,), dtype=tl.float32)
    tail_sqr = tl.zeros((BLOCK,), dtype=tl.float32)
    tail_count = tl.zeros((BLOCK,), dtype=tl.float32)
    for t in range((N - tail_start + BLOCK - 1) // BLOCK):
        col = offs + tail_start + t * BLOCK
        mask = col < N
        x = tl.load(X + row * N + col, mask=mask, other=0.0)
        tail_sum += tl.where(mask, x, 0.0)
        tail_sqr += tl.where(mask, x * x, 0.0)
        tail_count += tl.where(mask, 1.0, 0.0)
    # Reduce accumulators
    row_sum = tl.sum(sum_acc0, 0) + tl.sum(sum_acc1, 0) + tl.sum(tail_sum, 0)
    row_sqr = tl.sum(sqr_acc0, 0) + tl.sum(sqr_acc1, 0) + tl.sum(tail_sqr, 0)
    row_count = tl.sum(count_acc0, 0) + tl.sum(count_acc1, 0) + tl.sum(tail_count, 0)
    mean = tl.div_rn(row_sum, row_count)
    var = tl.div_rn(row_sqr, row_count) - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    # Main unrolled output loop
    for i in range(num_iters):
        base = 2 * i * BLOCK
        col0 = offs + base
        col1 = offs + base + BLOCK
        mask0 = col0 < N
        mask1 = col1 < N
        x0 = tl.load(X + row * N + col0, mask=mask0, other=0.0)
        x1 = tl.load(X + row * N + col1, mask=mask1, other=0.0)
        centered0 = x0 - mean
        centered1 = x1 - mean
        w0 = tl.load(W + col0, mask=mask0, other=0.0)
        w1 = tl.load(W + col1, mask=mask1, other=0.0)
        b0 = tl.load(B + col0, mask=mask0, other=0.0)
        b1 = tl.load(B + col1, mask=mask1, other=0.0)
        y0 = (centered0 * inv) * w0 + b0
        y1 = (centered1 * inv) * w1 + b1
        tl.store(Y + row * N + col0, y0, mask=mask0)
        tl.store(Y + row * N + col1, y1, mask=mask1)
    # Tail output
    for t in range((N - tail_start + BLOCK - 1) // BLOCK):
        col = offs + tail_start + t * BLOCK
        mask = col < N
        x = tl.load(X + row * N + col, mask=mask, other=0.0)
        centered = x - mean
        w = tl.load(W + col, mask=mask, other=0.0)
        b = tl.load(B + col, mask=mask, other=0.0)
        y = (centered * inv) * w + b
        tl.store(Y + row * N + col, y, mask=mask)
