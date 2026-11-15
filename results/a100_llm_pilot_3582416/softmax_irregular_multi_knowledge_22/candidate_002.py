import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    max_val = tl.max(x, 0)
    e = tl.exp(x - max_val)
    denom = tl.sum(e, 0)
    tl.store(Y + row * N + col, e / denom, col < N)
