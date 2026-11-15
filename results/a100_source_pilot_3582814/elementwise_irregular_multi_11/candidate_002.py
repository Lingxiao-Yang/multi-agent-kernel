import triton
import triton.language as tl

@triton.jit
def bias_add_relu_tiled(X, B, Y, N: tl.constexpr):
    BLOCK_SIZE = 32
    row = tl.program_id(0)  # row index
    block_id = tl.program_id(1)  # column block index
    col_start = block_id * BLOCK_SIZE
    offsets = col_start + tl.arange(0, BLOCK_SIZE)  # constexpr
    mask = offsets < N
    x_ptr = X + row * N + offsets
    b_ptr = B + offsets
    x = tl.load(x_ptr, mask, other=0.0)
    b = tl.load(b_ptr, mask, other=0.0)
    z = x + b
    y = tl.maximum(z, 0.0)
    tl.store(Y + row * N + offsets, y, mask)
