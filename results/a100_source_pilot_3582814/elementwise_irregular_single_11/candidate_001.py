import triton
import triton.language as tl

@triton.jit
def bias_add_relu(X, B, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    x = tl.load(X + offs, mask, other=0.0)
    b = tl.load(B + (offs % N), mask, other=0.0)
    r = x + b
    r = tl.maximum(r, 0.0)
    tl.store(Y + offs, r, mask)
