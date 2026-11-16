import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col_mask = offs < N
    x_ptr = X + row * N + offs
    x = tl.load(x_ptr, col_mask, other=-float('inf'))
    mx = tl.max(x, 0)
    x_shifted = x - mx
    exp_x = tl.exp(x_shifted)
    denom = tl.sum(exp_x, 0)
    softmax = exp_x / denom
    y_ptr = Y + row * N + offs
    tl.store(y_ptr, softmax, col_mask)
