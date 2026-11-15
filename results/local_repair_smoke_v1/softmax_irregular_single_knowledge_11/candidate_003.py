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

    # Compute max for the row using parallel reduction over BLOCK lanes
    mx = tl.max(x, 0)
    mx = tl.max_contiguous(mx, x)

    # Broadcast max to lanes
    mx_bcast = tl.broadcast_to(mx, BLOCK)

    # Compute exp(x - max) with masking
    exps = tl.where(mask, tl.exp(x - mx_bcast), 0.0)

    # Compute sum of exps using parallel reduction over BLOCK lanes
    s = tl.sum(exps, 0)
    denom = tl.sum(s)

    # Broadcast denom to lanes
    denom_bcast = tl.broadcast_to(denom, BLOCK)

    # Normalize and store output with masking
    y = tl.where(mask, exps / denom_bcast, 0.0)
    tl.store(Y + idx, y, mask)
