import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    # Load with masking to handle irregular dimensions
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    # Compute max for numeric stability
    mx = tl.max(x, 0)
    # Compute exponentials
    e = tl.exp(x - mx)
    # Sum exponentials
    s = tl.sum(e, 0)
    # Compute softmax output
    out = e / s
    # Store result with masking
    tl.store(Y + row * N + col, out, col < N)
