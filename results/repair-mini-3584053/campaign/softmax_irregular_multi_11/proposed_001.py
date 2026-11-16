import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # Each program processes one row
    col_block = tl.program_id(1)  # Tile block along columns
    offs = col_block * BLOCK + tl.arange(0, BLOCK)  # Column offsets

    mask = offs < N
    ptr = X + row * N + offs
    x = tl.load(ptr, mask, other=-float('inf'))

    # Compute max per tile
    mx = tl.max(x, axis=0)
    # Warp-level reduction for max across tiles
    mx = tl.max_contiguous(mx, axis=0)
    # Broadcast max across BLOCK lanes
    mx = tl.broadcast_to(mx, (BLOCK,))

    # Compute exp(x - max)
    shifted = tl.where(mask, tl.exp(x - mx), 0.0)

    # Compute sum per tile
    s = tl.sum(shifted, axis=0)
    # Warp-level reduction for sum across tiles
    s = tl.sum(s, axis=0)
    s = tl.broadcast_to(s, (BLOCK,))

    # Compute output
    out = tl.where(mask, shifted / s, 0.0)

    # Store output
    tl.store(Y + row * N + offs, out, mask)
