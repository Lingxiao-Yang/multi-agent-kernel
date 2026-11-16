import triton
import triton.language as tl

@triton.jit
def softmax_split_program_row(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    HALF = BLOCK // 2
    col = tl.arange(0, HALF)
    off = row * N + part * HALF + col
    mask = (part * HALF + col) < N
    # Load half-row
    x = tl.load(X + off, mask, other=-float('inf'))
    # Each program computes local max for its half
    local_max = tl.max(x, 0)
    # Write local max to shared memory (emulated using output buffer as workspace)
    max_ptr = Y + row * 4 + part
    tl.store(max_ptr, local_max)
    # Barrier (emulated): assume both maxes are written before continuing
    max0 = tl.load(Y + row * 4 + 0)
    max1 = tl.load(Y + row * 4 + 1)
    row_max = tl.maximum(max0, max1)
    # Center and exponentiate
    x_centered = x - row_max
    exp_x = tl.exp(x_centered)
    local_sum = tl.sum(exp_x, 0)
    sum_ptr = Y + row * 4 + (part + 2)
    tl.store(sum_ptr, local_sum)
    # Wait for both sum halves
    sum0 = tl.load(Y + row * 4 + 2)
    sum1 = tl.load(Y + row * 4 + 3)
    row_sum = sum0 + sum1
    y = exp_x / row_sum
    # Write output to Y after the 4 scratch slots
    y_off = row * N + part * HALF + col
    tl.store(Y + 513 * 4 + y_off, y, mask)
