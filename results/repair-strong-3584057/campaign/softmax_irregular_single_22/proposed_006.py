import triton
import triton.language as tl

@triton.jit
def softmax_fused_blockloop_split2(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    half = BLOCK // 2
    offs = tl.arange(0, half) + part * half
    mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    part_max = tl.max(x, 0)
    # Write partial max to scratch
    S_ptr = tl.make_block_ptr(
        base=tl.scratch, shape=(513, 2), strides=(2, 1), offsets=(row, part), block_shape=(1, 1), order=(0, 1), dtype=tl.float32
    )
    tl.store(S_ptr, part_max)
    tl.barrier()
    # Reduce to get full row max
    mx = tl.load(S_ptr, mask=True) if part == 0 else 0.0
    if part == 0:
        other_max = tl.load(S_ptr + 1, mask=True)
        mx = tl.maximum(mx, other_max)
        tl.store(S_ptr, mx)
    tl.barrier()
    mx = tl.load(S_ptr, mask=True)
    # Compute exp(x - mx) and partial denom
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    x_exp = tl.exp(x - mx)
    part_sum = tl.sum(x_exp, 0)
    tl.store(S_ptr, part_sum)
    tl.barrier()
    denom = tl.load(S_ptr, mask=True) if part == 0 else 0.0
    if part == 0:
        other_sum = tl.load(S_ptr + 1, mask=True)
        denom = denom + other_sum
        tl.store(S_ptr, denom)
    tl.barrier()
    denom = tl.load(S_ptr, mask=True)
    # Write output
    y = x_exp / denom
    y_ptrs = Y + row * N + offs
    tl.store(y_ptrs, y, mask=mask)
