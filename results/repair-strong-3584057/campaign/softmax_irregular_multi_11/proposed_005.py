import triton
import triton.language as tl

@triton.jit
def softmax_parallel_chunks(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    CHUNK_SIZE = BLOCK // 2  # 1024//2 = 512 for this case
    cols = tl.arange(0, 512) + chunk * 512
    mask = cols < N
    offs = row * N + cols
    neg_inf = -tl.float32(3.4028235e38)
    # Load current chunk
    x = tl.load(X + offs, mask, other=neg_inf)
    # Redundant: load both chunks to get row max
    chunk0_cols = tl.arange(0, 512)
    chunk0_mask = chunk0_cols < N
    chunk1_cols = tl.arange(0, 512) + 512
    chunk1_mask = chunk1_cols < N
    x0 = tl.load(X + row * N + chunk0_cols, chunk0_mask, other=neg_inf)
    x1 = tl.load(X + row * N + chunk1_cols, chunk1_mask, other=neg_inf)
    mx0 = tl.max(x0, 0)
    mx1 = tl.max(x1, 0)
    mx = tl.maximum(mx0, mx1)
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    # Redundant: sum over both chunks for correct denom
    exp_x0 = tl.exp(x0 - mx)
    exp_x1 = tl.exp(x1 - mx)
    sum0 = tl.sum(exp_x0, 0)
    sum1 = tl.sum(exp_x1, 0)
    denom = sum0 + sum1
    y = exp_x / denom
    tl.store(Y + offs, y, mask)
