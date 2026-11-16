import triton
import triton.language as tl

@triton.jit
def fused_softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs
    x_ptrs = X + row * N + col
    mask = col < N
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    mx = tl.max(x, 0)
    x_centered = x - mx
    numerators = tl.exp(x_centered)
    denom = tl.sum(numerators, 0)
    softmax = numerators / denom
    tl.store(Y + row * N + col, softmax, mask=mask)
