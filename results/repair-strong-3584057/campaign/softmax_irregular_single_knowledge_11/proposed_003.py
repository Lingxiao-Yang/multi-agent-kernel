import triton
import triton.language as tl

@triton.jit
def fused_softmax_rowblock(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    num_blocks = (N + BLOCK - 1) // BLOCK
    col_start = block_id * (N // 2)
    col = col_start + offs
    mask = col < N
    # 1. Compute row max (across all columns, blockwise)
    max_val = -float('inf')
    for b in range(num_blocks):
        blk_col = b * BLOCK + offs
        blk_mask = blk_col < N
        x_ptrs = X + row * N + blk_col
        x = tl.load(x_ptrs, mask=blk_mask, other=-float('inf'))
        blk_max = tl.max(x, 0)
        max_val = tl.maximum(max_val, blk_max)
    max_val = tl.max(max_val, 0)
    # 2. Compute denominator (sum of exp(x - max))
    denom = 0.0
    for b in range(num_blocks):
        blk_col = b * BLOCK + offs
        blk_mask = blk_col < N
        x_ptrs = X + row * N + blk_col
        x = tl.load(x_ptrs, mask=blk_mask, other=-float('inf'))
        exp_x = tl.exp(x - max_val)
        blk_sum = tl.sum(exp_x, 0)
        denom = denom + blk_sum
    # 3. Compute and store the softmax for this block
    x_ptrs = X + row * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    exp_x = tl.exp(x - max_val)
    softmax = exp_x / denom
    y_ptrs = Y + row * N + col
    tl.store(y_ptrs, softmax, mask=mask)
