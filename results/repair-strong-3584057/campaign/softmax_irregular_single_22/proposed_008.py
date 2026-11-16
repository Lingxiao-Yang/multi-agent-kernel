import triton
import triton.language as tl

@triton.jit
def softmax_split_chunks(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    half = (BLOCK + 1) // 2  # Split BLOCK into 2 nearly equal parts
    offs = tl.arange(0, half)
    base = part * half
    idx = base + offs
    mask = idx < N
    x_ptrs = X + row * N + idx
    # Phase 1: Local max
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    local_max = tl.max(x, 0)
    # Write local max to scratch
    max_scratch_ptr = tl.scratch + row * 2 + part
    tl.store(max_scratch_ptr, local_max)
    tl.barrier()
    # Reduce to row max
    if part == 0:
        max0 = tl.load(tl.scratch + row * 2 + 0)
        max1 = tl.load(tl.scratch + row * 2 + 1)
        row_max = tl.maximum(max0, max1)
        # Store back to both slots for both parts
        tl.store(tl.scratch + row * 2 + 0, row_max)
        tl.store(tl.scratch + row * 2 + 1, row_max)
    tl.barrier()
    row_max = tl.load(tl.scratch + row * 2 + part)
    # Phase 2: Local exp-sum
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    exp_x = tl.exp(x - row_max)
    local_sum = tl.sum(exp_x, 0)
    sum_scratch_ptr = tl.scratch + (513 * 2) + row * 2 + part
    tl.store(sum_scratch_ptr, local_sum)
    tl.barrier()
    # Reduce to row sum
    if part == 0:
        sum0 = tl.load(tl.scratch + (513 * 2) + row * 2 + 0)
        sum1 = tl.load(tl.scratch + (513 * 2) + row * 2 + 1)
        row_sum = sum0 + sum1
        tl.store(tl.scratch + (513 * 2) + row * 2 + 0, row_sum)
        tl.store(tl.scratch + (513 * 2) + row * 2 + 1, row_sum)
    tl.barrier()
    row_sum = tl.load(tl.scratch + (513 * 2) + row * 2 + part)
    # Phase 3: Write normalized outputs
    y = exp_x / row_sum
    y_ptrs = Y + row * N + idx
    tl.store(y_ptrs, y, mask=mask)
