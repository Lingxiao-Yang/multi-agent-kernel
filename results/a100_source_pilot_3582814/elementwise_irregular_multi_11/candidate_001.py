import triton
import triton.language as tl

@triton.jit
def bias_add_relu_tiled(X, B, Y, N: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    block_size = 32
    col_start = block_id * block_size
    offsets = col_start + tl.arange(0, block_size)
    mask = offsets < N
    x = tl.load(X + row * N + offsets, mask, other=0.0)
    b = tl.load(B + offsets, mask, other=0.0)
    z = x + b
    y = tl.maximum(z, 0.0)
    tl.store(Y + row * N + offsets, y, mask)

@triton.jit
def relu_tail(Y, N: tl.constexpr):
    row = tl.program_id(0)
    col = 800  # Offset beyond tiled blocks
    mask = col < N
    x = tl.load(Y + row * N + col, mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y + row * N + col, y, mask)
