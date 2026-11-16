import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    col = offs
    mask = col < N
    x_ptrs = X + row_id * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    mx = tl.max(x, 0)
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    denom = tl.sum(exp_x, 0)
    norm = exp_x / denom
    y_ptrs = Y + row_id * N + col
    tl.store(y_ptrs, norm, mask=mask)
