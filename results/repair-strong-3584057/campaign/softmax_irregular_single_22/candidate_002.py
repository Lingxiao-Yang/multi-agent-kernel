import triton
import triton.language as tl

@triton.jit
def softmax_split_reduce(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    chunk = BLOCK // 2
    offs = tl.arange(0, chunk) + part * chunk
    mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    # Partial reduction: max per chunk
    mx_local = tl.max(x, 0)
    # Reduce max across parts (within row)
    mx = tl.maximum(
        tl.max_contiguous(mx_local, 0),
        tl.max_contiguous(mx_local, 0)
    )
    # Broadcast mx for exp
    x_exp = tl.exp(x - mx)
    sum_local = tl.sum(x_exp, 0)
    # Reduce sum across parts
    sum_exp = tl.sum(tl.broadcast_to(sum_local, (chunk,)), 0)
    # Normalize
    y = x_exp / sum_exp
    y_ptrs = Y + row * N + offs
    tl.store(y_ptrs, y, mask=mask)
