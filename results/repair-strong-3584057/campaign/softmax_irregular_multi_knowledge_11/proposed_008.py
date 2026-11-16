import triton
import triton.language as tl

@triton.jit
def softmax_split_program_row(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, HALF: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    col = tl.arange(0, HALF)
    base = row * N + part * HALF
    off = base + col
    mask = (part * HALF + col) < N
    # Load half-row
    x = tl.load(X + off, mask=mask, other=-float('inf'))
    # Each program computes local max for its half
    local_max = tl.max(x, 0)
    # Write local max to output buffer as workspace: first 4 floats per row
    # layout: [max0, max1, sum0, sum1] for each row
    max_ptr = Y + row * 4 + part
    tl.store(max_ptr, local_max)
    # Both programs read local maxes to compute full row max
    max0 = tl.load(Y + row * 4 + 0)
    max1 = tl.load(Y + row * 4 + 1)
    row_max = tl.maximum(max0, max1)
    # Center and exponentiate
    x_centered = x - row_max
    exp_x = tl.exp(x_centered)
    local_sum = tl.sum(exp_x, 0)
    sum_ptr = Y + row * 4 + (part + 2)
    tl.store(sum_ptr, local_sum)
    # Both programs read local sums for normalization
    sum0 = tl.load(Y + row * 4 + 2)
    sum1 = tl.load(Y + row * 4 + 3)
    row_sum = sum0 + sum1
    y = exp_x / row_sum
    # Write output after workspace region (offset 513*4 floats)
    out_base = 513 * 4
    y_off = out_base + row * N + part * HALF + col
    tl.store(Y + y_off, y, mask=mask)
