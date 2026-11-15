import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load inputs with masking; out-of-bounds loads get -inf for max reduction
    x = tl.load(X + row * N + offs, mask, other=-float('inf'))

    # Compute max per row for numerical stability
    mx = tl.max(x, axis=0)

    # Compute exponentials with masked padding set to zero
    exps = tl.where(mask, tl.exp(x - mx), 0.0)

    # Sum of exponentials per row
    denom = tl.sum(exps, axis=0)

    # Compute normalized softmax output with masking
    y = tl.where(mask, exps / denom, 0.0)

    # Store output with masking
    tl.store(Y + row * N + offs, y, mask)
