import triton
import triton.language as tl

@triton.jit
def softmax_row_col_parallel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr, SPLIT: tl.constexpr):
    row_id = tl.program_id(0)
    part = tl.program_id(1)
    offs = tl.arange(0, BLOCK)
    tile_cols = (N + SPLIT - 1) // SPLIT
    col_start = part * tile_cols
    col = col_start + offs
    mask = col < N
    x_ptrs = X + row_id * N + col
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    local_max = tl.max(x, 0)
    smem_max = tl.zeros((SPLIT,), dtype=tl.float32)
    tl.store(smem_max + part, local_max)
    tl.barrier()
    if offs[0] == 0:
        row_max = smem_max[0]
        for i in range(1, SPLIT):
            row_max = tl.maximum(row_max, smem_max[i])
        smem_max[0] = row_max
    tl.barrier()
    row_max = smem_max[0]
    x_centered = x - row_max
    exp_x = tl.where(mask, tl.exp(x_centered), 0.0)
    local_sum = tl.sum(exp_x, 0)
    smem_sum = tl.zeros((SPLIT,), dtype=tl.float32)
    tl.store(smem_sum + part, local_sum)
    tl.barrier()
    if offs[0] == 0:
        denom = smem_sum[0]
        for i in range(1, SPLIT):
            denom = denom + smem_sum[i]
        smem_sum[0] = denom
    tl.barrier()
    denom = smem_sum[0]
    norm = exp_x / denom
    y_ptrs = Y + row_id * N + col
    tl.store(y_ptrs, norm, mask=mask)
