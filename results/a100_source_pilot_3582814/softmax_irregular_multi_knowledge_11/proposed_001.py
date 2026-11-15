import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    col = block_id * BLOCK + tl.arange(0, BLOCK)

    mask = col < N
    offs = row * N + col

    # Load input with masking for boundary
    x = tl.load(X + offs, mask, other=-float('inf'))

    # Compute max per row block (reduction over BLOCK dimension)
    mx = tl.max(x, axis=0)

    # Compute exp and sum for normalization
    exps = tl.exp(x - mx)
    denom = tl.sum(exps, axis=0)

    # Normalize
    y = exps / denom

    # Store output with masking
    tl.store(Y + offs, y, mask)
