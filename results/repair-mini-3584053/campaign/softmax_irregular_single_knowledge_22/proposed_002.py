import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    idx = row * N + offs

    # Load input with masking, pad out-of-bounds with -inf for max reduction
    x = tl.load(X + idx, col_mask, other=-float('inf'))

    # Two-stage max reduction: split BLOCK into two halves
    half = BLOCK // 2
    max_left = tl.max(x[:half], col_mask[:half])
    max_right = tl.max(x[half:], col_mask[half:])
    mx = tl.maximum(max_left, max_right)

    # Broadcast max to all lanes
    mx_broadcast = tl.broadcast_to(mx, (BLOCK,))

    # Compute exp(x - max) with masking
    exp_x = tl.exp(x - mx_broadcast) * col_mask

    # Two-stage sum reduction for denominator
    sum_left = tl.sum(exp_x[:half], col_mask[:half])
    sum_right = tl.sum(exp_x[half:], col_mask[half:])
    denom = sum_left + sum_right

    # Broadcast denom to all lanes
    denom_broadcast = tl.broadcast_to(denom, (BLOCK,))

    # Compute output
    out = exp_x / denom_broadcast

    # Store output with masking
    tl.store(Y + idx, out, col_mask)
