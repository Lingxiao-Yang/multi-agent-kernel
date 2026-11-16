import triton
import triton.language as tl

@triton.jit
def softmax_persistent_blockwise(X, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    rows_per_program = (M + tl.num_programs(0) - 1) // tl.num_programs(0)
    row_start = pid * rows_per_program
    row_end = min(row_start + rows_per_program, M)
    for row_id in range(row_start, row_end):
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
