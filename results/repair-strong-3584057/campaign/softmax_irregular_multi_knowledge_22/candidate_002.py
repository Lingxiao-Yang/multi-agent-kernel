import triton
import triton.language as tl

@triton.jit
def softmax_persistent(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    # Persistent kernel: single program processes all rows
    offs = tl.arange(0, BLOCK)
    for row_id in range(M):
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
