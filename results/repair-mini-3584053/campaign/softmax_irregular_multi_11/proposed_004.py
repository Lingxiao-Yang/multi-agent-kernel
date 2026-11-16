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

    # Warp-level max reduction across all lanes
    mx = tl.max_contiguous(x)

    # Broadcast max scalar to all lanes
    mx_bcast = tl.broadcast_to(mx, (BLOCK,))

    # Compute exp(x - max) with masking
    shifted = tl.where(mask, tl.exp(x - mx_bcast), 0.0)

    # Warp-level sum reduction across all lanes
    s = tl.sum(shifted)

    # Broadcast sum scalar to all lanes
    s_bcast = tl.broadcast_to(s, (BLOCK,))

    # Compute normalized output
    out = tl.where(mask, shifted / s_bcast, 0.0)

    # Store output with masking
    tl.store(Y + row * N + offs, out, mask)
