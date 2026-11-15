import triton
import triton.language as tl

@triton.jit
def bias_add(X, B, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < TOTAL, other=0)
    b = tl.load(B + i % N, i < TOTAL, other=0)
    tl.store(Y + i, x + b, i < TOTAL)

@triton.jit
def relu(X, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < TOTAL, other=0)
    tl.store(Y + i, tl.maximum(x, 0), i < TOTAL)
