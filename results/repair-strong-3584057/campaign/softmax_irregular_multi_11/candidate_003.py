import triton
import triton.language as tl

@triton.jit
def softmax_parallel_chunks(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    CHUNK_SIZE = BLOCK // 2
    cols = tl.arange(0, CHUNK_SIZE) + chunk * CHUNK_SIZE
    mask = cols < N
    offs = row * N + cols
    # Use minimum float32 value as -inf identity
    neg_inf = -tl.float32(3.4028235e38)
    x = tl.load(X + offs, mask, other=neg_inf)
    # Each chunk computes local max
    local_mx = tl.max(x, 0)
    # Write local max to shared buffer (explicitly in registers)
    mx_buf = tl.zeros((2,), dtype=tl.float32)
    mx_buf = tl.where(chunk == 0, tl.broadcast_to(local_mx, (2,)), mx_buf)
    mx_buf = tl.where(chunk == 1, tl.broadcast_to(local_mx, (2,)), mx_buf)
    # Reduce max across both chunks
    mx = tl.max(mx_buf, 0)
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    # Each chunk computes local sum
    local_sum = tl.sum(exp_x, 0)
    sum_buf = tl.zeros((2,), dtype=tl.float32)
    sum_buf = tl.where(chunk == 0, tl.broadcast_to(local_sum, (2,)), sum_buf)
    sum_buf = tl.where(chunk == 1, tl.broadcast_to(local_sum, (2,)), sum_buf)
    denom = tl.sum(sum_buf, 0)
    y = exp_x / denom
    tl.store(Y + offs, y, mask)
