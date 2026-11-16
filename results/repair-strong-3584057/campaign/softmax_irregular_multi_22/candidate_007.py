import triton
import triton.language as tl

@triton.jit
def softmax_split_col_unrolled(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    tile = RB // 2
    # Unroll both halves explicitly
    offs0 = tl.arange(0, tile)
    cols0 = offs0
    col_mask0 = cols0 < N
    x_ptr0 = X + row * N + cols0
    x0 = tl.load(x_ptr0, col_mask0, other=-float('inf'))
    offs1 = tl.arange(0, tile)
    cols1 = tile + offs1
    col_mask1 = cols1 < N
    x_ptr1 = X + row * N + cols1
    x1 = tl.load(x_ptr1, col_mask1, other=-float('inf'))
    # Compute the global max over both halves
    mx0 = tl.max(x0, 0)
    mx1 = tl.max(x1, 0)
    mx = tl.maximum(mx0, mx1)
    # Compute exponentials and sums for both halves
    x0_shifted = x0 - mx
    x1_shifted = x1 - mx
    exp_x0 = tl.exp(x0_shifted)
    exp_x1 = tl.exp(x1_shifted)
    sum0 = tl.sum(exp_x0, 0)
    sum1 = tl.sum(exp_x1, 0)
    denom = sum0 + sum1
    # Each program writes its own tile only
    if part == 0:
        softmax = exp_x0 / denom
        tl.store(Y + row * N + cols0, softmax, col_mask0)
    else:
        softmax = exp_x1 / denom
        tl.store(Y + row * N + cols1, softmax, col_mask1)
