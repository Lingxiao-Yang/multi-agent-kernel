import triton
import triton.language as tl

@triton.jit
def fused_softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs
    idx = row * N + col
    mask = col < N

    # Load input with mask; out-of-bound elements set to -inf for max reduction
    x = tl.load(X + idx, mask, other=-float('inf'))

    # Compute max for the row using parallel reduction
    mx = tl.max(x, axis=0)
    mx = tl.max_contiguous(mx)

    # Compute exp(x - max) with masking
    exps = tl.where(mask, tl.exp(x - mx), 0.0)
    s = tl.sum(exps, axis=0)
    denom = tl.sum(s)

    # Normalize and store output with masking
    y = tl.where(mask, exps / denom, 0.0)
    tl.store(Y + idx, y, mask)
