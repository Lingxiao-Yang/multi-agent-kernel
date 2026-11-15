import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, TOTAL: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, TOTAL)
    col = offs
    # Load elements of the row with mask for boundary
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    # Compute max for numeric stability
    mx = tl.max(x, 0)
    # Subtract max, exponentiate
    exps = tl.exp(x - mx)
    # Reduce sum of exps
    denom = tl.sum(exps, 0)
    # Compute softmax output
    out = exps / denom
    # Store result with mask
    tl.store(Y + row * N + col, out, col < N)
