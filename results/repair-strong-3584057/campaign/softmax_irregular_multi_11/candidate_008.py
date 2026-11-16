import triton
import triton.language as tl

@triton.jit
def softmax_blockwise(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    # Each block covers 512 columns
    cols = tl.arange(0, 512) + block_id * 512
    mask = cols < N
    offs = row * N + cols
    neg_inf = -3.4028235e38
    # Load this block's data
    x_block = tl.load(X + offs, mask, other=neg_inf)
    # --- Blockwise max reduction (across both blocks) ---
    # Each block loads both chunks for the row
    block0_cols = tl.arange(0, 512)
    block0_mask = block0_cols < N
    block1_cols = tl.arange(0, 512) + 512
    block1_mask = block1_cols < N
    x0 = tl.load(X + row * N + block0_cols, block0_mask, other=neg_inf)
    x1 = tl.load(X + row * N + block1_cols, block1_mask, other=neg_inf)
    mx0 = tl.max(x0, 0)
    mx1 = tl.max(x1, 0)
    mx = tl.maximum(mx0, mx1)
    # --- Center, exp, compute denominator (blockwise sum) ---
    x_block_centered = x_block - mx
    exp_x_block = tl.exp(x_block_centered)
    # Redundant: both blocks compute blockwise sums for full row
    exp_x0 = tl.exp(x0 - mx)
    exp_x1 = tl.exp(x1 - mx)
    sum0 = tl.sum(exp_x0, 0)
    sum1 = tl.sum(exp_x1, 0)
    denom = sum0 + sum1
    # --- Normalize and store ---
    y = exp_x_block / denom
    tl.store(Y + offs, y, mask)
