import triton
import triton.language as tl

@triton.jit
def fused_softmax_splitrow(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    # Each program covers half a row: [row, half]
    pid = tl.program_id(0)
    row = pid // 2
    half = pid % 2
    offs = tl.arange(0, BLOCK)
    half_cols = (N + 1) // 2
    col_start = half * half_cols
    col = col_start + offs
    mask = (row < 513) & (col < N)
    # 1. Compute global row max (scalar)
    max_val = -float('inf')
    for h in range(2):
        h_col_start = h * half_cols
        h_col = h_col_start + offs
        h_mask = (row < 513) & (h_col < N)
        ptr = X + row * N + h_col
        x = tl.load(ptr, mask=h_mask, other=-float('inf'))
        block_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, block_max)
    # 2. Compute denominator (sum of exp(x - max)) (scalar)
    denom = 0.0
    for h in range(2):
        h_col_start = h * half_cols
        h_col = h_col_start + offs
        h_mask = (row < 513) & (h_col < N)
        ptr = X + row * N + h_col
        x = tl.load(ptr, mask=h_mask, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        block_sum = tl.sum(exp_x, 0)
        denom = denom + block_sum
    # 3. Write output for this half-row
    ptr = X + row * N + col
    x = tl.load(ptr, mask=mask, other=-float('inf'))
    exp_x = tl.exp(x - max_val)
    softmax = exp_x / denom
    out_ptr = Y + row * N + col
    tl.store(out_ptr, softmax, mask=mask)
