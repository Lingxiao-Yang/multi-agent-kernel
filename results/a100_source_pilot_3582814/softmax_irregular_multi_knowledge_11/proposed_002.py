import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    offs_in_block = tl.arange(0, BLOCK)
    col = block_id * BLOCK + offs_in_block
    mask = col < N
    offs = row * N + col

    # Load input with masking for boundary elements
    x = tl.load(X + offs, mask, other=-float('inf'))

    # Compute partial max per block
    block_max = tl.max(x, axis=0)

    # Compute exponentials shifted by block max
    exps = tl.exp(x - block_max)

    # Store partial exponentials for final reduction
    tl.store(Y + offs, exps, mask)

@triton.jit
def softmax_fused_final_reduce(Y, N: tl.constexpr):
    row = tl.program_id(0)
    offs_in_row = tl.arange(0, N)
    offs = row * N + offs_in_row
    mask = offs_in_row < N

    # Load partial exponentials
    vals = tl.load(Y + offs, mask, other=0.0)

    # Compute max of partial exponentials for numerical stability
    mx = tl.max(vals, axis=0)

    # Recompute exp shifted by global max for stable sum
    exps = tl.exp(vals - mx)

    # Compute sum of exponentials
    denom = tl.sum(exps, axis=0)

    # Normalize
    y = exps / denom

    # Store normalized output
    tl.store(Y + offs, y, mask)
