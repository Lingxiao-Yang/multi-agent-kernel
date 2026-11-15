import triton
import triton.language as tl

@triton.jit
def softmax_stats_stage1(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    offs = col_block * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # Load slice with padding -inf for max reduction
    x = tl.load(X + row * N + offs, mask, other=-float('inf'))
    local_max = tl.max(x, 0)

    # Subtract local max for numerical stability and compute exp and sum
    exp_x = tl.exp(x - local_max)
    local_denom = tl.sum(exp_x, 0)

    # Store partial max and denom per block
    # layout: S[row, col_block*2] = max, S[row, col_block*2+1] = denom
    ptr_max = S + row * 8 + col_block * 2
    ptr_denom = ptr_max + 1
    tl.store(ptr_max, local_max)
    tl.store(ptr_denom, local_denom)

@triton.jit
def softmax_stats_stage2(S_in, S_out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)

    # Load all partial max and denom from stage1 (BLOCK=4 blocks, each block 2 elements)
    offs = tl.arange(0, BLOCK * 2)  # 8 elements: max, denom pairs
    mask = offs < BLOCK * 2
    vals = tl.load(S_in + row * 8 + offs, mask, other=-float('inf'))

    # Separate max and denom values
    max_vals = tl.where(offs % 2 == 0, vals, -float('inf'))
    denom_vals = tl.where(offs % 2 == 1, vals, 0.0)

    # Reduce max over blocks
    max_val = tl.max(max_vals, 0)

    # Adjust denom sums by exp(max_vals - max_val) to combine partial sums correctly
    adj = tl.exp(tl.where(max_vals != -float('inf'), max_vals - max_val, -float('inf')))
    denom_val = tl.sum(denom_vals * adj, 0)

    # Store global max and denom
    tl.store(S_out + row * 2, max_val)
    tl.store(S_out + row * 2 + 1, denom_val)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = idx // N
    mask = idx < TOTAL

    x = tl.load(X + idx, mask, other=0.0)
    max_val = tl.load(S + row * 2, mask, other=0.0)
    denom = tl.load(S + row * 2 + 1, mask, other=1.0)

    y = tl.exp(x - max_val) / denom
    tl.store(Y + idx, y, mask)
