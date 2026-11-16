import triton
import triton.language as tl

@triton.jit
def softmax_fused_blockwise(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # Blockwise reduction for row max
    mx = -float('inf')
    for start in range(0, N, BLOCK):
        col = start + offs
        mask = col < N
        x_ptrs = X + row_id * N + col
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        mx = tl.maximum(mx, tl.max(x, 0))
    # Blockwise sum of exp(x - mx)
    denom = 0.0
    for start in range(0, N, BLOCK):
        col = start + offs
        mask = col < N
        x_ptrs = X + row_id * N + col
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        exp_x = tl.exp(x - mx)
        # Masked out-of-bounds columns contribute 0
        exp_x = tl.where(mask, exp_x, 0.0)
        denom = denom + tl.sum(exp_x, 0)
    # Blockwise write softmax output
    for start in range(0, N, BLOCK):
        col = start + offs
        mask = col < N
        x_ptrs = X + row_id * N + col
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        exp_x = tl.exp(x - mx)
        norm = exp_x / denom
        y_ptrs = Y + row_id * N + col
        tl.store(y_ptrs, norm, mask=mask)
