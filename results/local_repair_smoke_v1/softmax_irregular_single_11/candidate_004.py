import triton
import triton.language as tl

@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    block_id = tl.program_id(1)
    offs = block_id * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs
    x = tl.load(ptr, mask, other=-float('inf'))
    mx = tl.max(x, 0)
    s_ptr = S + row * ((N + BLOCK - 1) // BLOCK * 2) + block_id * 2
    tl.store(s_ptr, mx)
    x_exp = tl.where(mask, tl.exp(x - mx), 0.0)
    denom = tl.sum(x_exp, 0)
    tl.store(s_ptr + 1, denom)

@triton.jit
def softmax_stats_finalize(S, N: tl.constexpr):
    row = tl.program_id(0)
    num_blocks = (N + 256 - 1) // 256
    mx = -float('inf')
    for i in range(num_blocks):
        mx_block = tl.load(S + row * num_blocks * 2 + i * 2)
        mx = tl.maximum(mx, mx_block)
    denom = 0.0
    for i in range(num_blocks):
        val = tl.load(S + row * num_blocks * 2 + i * 2 + 1)
        denom += tl.exp(val - mx)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)

@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    row = i // N
    x = tl.load(X + i, mask, other=0.0)
    mx = tl.load(S + row * 2, mask, other=-float('inf'))
    denom = tl.load(S + row * 2 + 1, mask, other=1.0)
    out = tl.exp(x - mx) / denom
    tl.store(Y + i, out, mask)
