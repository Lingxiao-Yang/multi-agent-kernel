import triton
import triton.language as tl

@triton.jit
def softmax_split_reduce_scratch(X, Y, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    half = BLOCK // 2
    offs = tl.arange(0, half) + part * half
    mask = offs < N
    # Load input chunk
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    # Compute local max
    mx_local = tl.max(x, 0)
    # Store local max to scratch
    S[row, part] = mx_local
    tl.barrier()
    # Compute row max by reducing across both parts
    mx_row = tl.maximum(S[row, 0], S[row, 1])
    # Exponentiate centered
    x_exp = tl.exp(x - mx_row)
    # Compute local exp sum
    exp_sum_local = tl.sum(x_exp, 0)
    # Store local exp sum to scratch
    S[row, part] = exp_sum_local
    tl.barrier()
    # Compute row exp sum by reducing across both parts
    exp_sum_row = S[row, 0] + S[row, 1]
    # Normalize
    y = x_exp / exp_sum_row
    # Store output
    y_ptrs = Y + row * N + offs
    tl.store(y_ptrs, y, mask=mask)
