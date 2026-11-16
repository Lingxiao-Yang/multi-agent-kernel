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

    # Two-stage max reduction: split BLOCK into two halves with masking
    half = BLOCK // 2
    left_mask = offs < half
    right_mask = ~left_mask

    # Mask values outside half for max reduction
    x_left = tl.where(left_mask, x, -float('inf'))
    x_right = tl.where(right_mask, x, -float('inf'))
    max_left = tl.max(x_left, 0)
    max_right = tl.max(x_right, 0)
    mx = tl.maximum(max_left, max_right)

    # Broadcast max to all lanes
    mx_broadcast = tl.broadcast_to(mx, (BLOCK,))

    # Compute exp(x - max) with masking
    exp_x = tl.exp(x - mx_broadcast) * col_mask

    # Two-stage sum reduction for denominator with masking
    exp_left = tl.where(left_mask, exp_x, 0.0)
    exp_right = tl.where(right_mask, exp_x, 0.0)
    sum_left = tl.sum(exp_left, 0)
    sum_right = tl.sum(exp_right, 0)
    denom = sum_left + sum_right

    # Broadcast denom to all lanes
    denom_broadcast = tl.broadcast_to(denom, (BLOCK,))

    # Compute output
    out = exp_x / denom_broadcast

    # Store output with masking
    tl.store(Y + idx, out, col_mask)
