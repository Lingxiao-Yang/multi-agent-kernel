import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    mx = tl.max(x, 0)
    x_exp = tl.exp(x - mx)
    denom = tl.sum(x_exp, 0)
    y = x_exp / denom
    y_ptrs = Y + row * N + offs
    tl.store(y_ptrs, y, mask=mask)
