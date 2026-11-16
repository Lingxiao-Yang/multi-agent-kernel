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

    # Compute max per row reduction
    mx = tl.max(x, 0)

    # Compute exp(x - max) with masked values masked to zero
    exp_x = tl.exp(x - mx) * col_mask

    # Sum exp values for denominator
    denom = tl.sum(exp_x, 0)

    # Compute softmax output
    out = exp_x / denom

    # Store output with masking
    tl.store(Y + idx, out, col_mask)
