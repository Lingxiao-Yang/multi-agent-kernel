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
    mx_tile = tl.max(x)
    # Warp-level max reduction across all lanes
    mx = tl.max_contiguous(mx_tile)

    # Broadcast max scalar to BLOCK lanes
    mx_broadcast = tl.broadcast_to(mx, (BLOCK,))

    # Compute exp(x - max)
    shifted = tl.where(mask, tl.exp(x - mx_broadcast), 0.0)

    # Compute sum per tile
    s_tile = tl.sum(shifted)
    # Warp-level sum reduction across all lanes
    s = tl.sum(s_tile)

    # Broadcast sum scalar to BLOCK lanes
    s_broadcast = tl.broadcast_to(s, (BLOCK,))

    # Compute output
    out = tl.where(mask, shifted / s_broadcast, 0.0)

    # Store output
    tl.store(Y + row * N + offs, out, mask)
