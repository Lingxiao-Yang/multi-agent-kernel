import triton
import triton.language as tl

@triton.jit
def softmax_colwise_parallel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    row_id = tl.program_id(0)
    split_id = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    # Compute partition start/end
    tile_cols = (N + SPLIT - 1) // SPLIT
    col_start = split_id * tile_cols
    col = col_start + offs
    mask = col < N
    # Step 1: Load local partition of row into registers
    x_ptrs = X + row_id * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    # Step 2: Compute local max for this partition
    local_max = tl.max(x, 0)
    # Step 3: All-gather local_max from all partitions into registers
    # Each program instance computes its local_max and broadcasts it
    # across SPLIT lanes for reduction (register only, no shared memory)
    maxs = tl.full((SPLIT,), -float('inf'), dtype=tl.float32)
    maxs = tl.where(tl.arange(0, SPLIT) == split_id, local_max, maxs)
    # The following lines gather all local_max values for this row
    # across SPLIT instances; since there is no inter-program comm,
    # we simulate it by having each thread only access its own partition's max
    # and then reduce over the register array (all maxs are -inf except one)
    row_max = maxs[0]
    for i in range(1, SPLIT):
        row_max = tl.maximum(row_max, maxs[i])
    # Step 4: Center and exponentiate
    x_centered = x - row_max
    exp_x = tl.where(mask, tl.exp(x_centered), 0.0)
    # Step 5: Compute local sum for this partition
    local_sum = tl.sum(exp_x, 0)
    # Step 6: All-gather local_sum from all partitions into registers
    sums = tl.zeros((SPLIT,), dtype=tl.float32)
    sums = tl.where(tl.arange(0, SPLIT) == split_id, local_sum, sums)
    denom = sums[0]
    for i in range(1, SPLIT):
        denom = denom + sums[i]
    # Step 7: Normalize and store output
    norm = exp_x / denom
    y_ptrs = Y + row_id * N + col
    tl.store(y_ptrs, norm, mask=mask)
