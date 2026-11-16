import triton
import triton.language as tl

@triton.jit
def fused_softmax_blockwise(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    col_start = block * BLOCK
    col = col_start + offs
    mask = col < N
    num_blocks = (N + BLOCK - 1) // BLOCK
    # Compute local max for this block
    x_ptrs = X + row * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    local_max = tl.max(x, 0)
    # Reduce for global row max across all blocks
    max_val = local_max
    for b in range(num_blocks):
        blk_col = b * BLOCK + offs
        blk_mask = blk_col < N
        blk_ptrs = X + row * N + blk_col
        blk_x = tl.load(blk_ptrs, mask=blk_mask, other=-float('inf'))
        blk_max = tl.max(blk_x, 0)
        max_val = tl.maximum(max_val, blk_max)
    # Broadcast max_val within block
    max_val = tl.max(max_val, 0)
    # Compute local exp(x - max)
    x_centered = x - max_val
    exp_x = tl.exp(x_centered)
    local_sum = tl.sum(exp_x, 0)
    # Reduce for global denominator across all blocks
    denom = local_sum
    for b in range(num_blocks):
        blk_col = b * BLOCK + offs
        blk_mask = blk_col < N
        blk_ptrs = X + row * N + blk_col
        blk_x = tl.load(blk_ptrs, mask=blk_mask, other=-float('inf'))
        blk_exp = tl.exp(blk_x - max_val)
        blk_sum = tl.sum(blk_exp, 0)
        denom += blk_sum
    # Write softmax output for this block
    softmax = exp_x / denom
    y_ptrs = Y + row * N + col
    tl.store(y_ptrs, softmax, mask=mask)
