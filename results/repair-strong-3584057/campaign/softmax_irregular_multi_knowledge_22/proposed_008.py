import triton
import triton.language as tl

@triton.jit
def softmax_row_split(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    split_id = tl.program_id(1)
    SPLIT = 2  # Number of splits across columns
    chunk = (N + SPLIT - 1) // SPLIT
    col_start = split_id * chunk
    col_end = min(col_start + chunk, N)
    offs = tl.arange(0, BLOCK)
    col = col_start + offs
    mask = (col < col_end) & (col < N)
    x_ptrs = X + row_id * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    # Local max for this partition
    local_max = tl.max(x, 0)
    # Gather both splits' local_max and compute row_max by arithmetic
    max0 = tl.max(tl.where(split_id == 0, local_max, -float('inf')), 0)
    max1 = tl.max(tl.where(split_id == 1, local_max, -float('inf')), 0)
    row_max = tl.maximum(max0, max1)
    x_centered = x - row_max
    exp_x = tl.exp(x_centered)
    local_sum = tl.sum(exp_x, 0)
    sum0 = tl.sum(tl.where(split_id == 0, local_sum, 0.0), 0)
    sum1 = tl.sum(tl.where(split_id == 1, local_sum, 0.0), 0)
    denom = sum0 + sum1
    norm = exp_x / denom
    y_ptrs = Y + row_id * N + col
    tl.store(y_ptrs, norm, mask=mask)
