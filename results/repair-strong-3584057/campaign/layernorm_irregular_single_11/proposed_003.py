import triton
import triton.language as tl

@triton.jit
def layernorm_stats_rowwise(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x_ptrs = X + row * N + offs
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    sum_x = tl.sum(x, axis=0)
    mean = tl.div_rn(sum_x, N)
    centered = tl.where(mask, x - mean, 0.0)
    sum_centered2 = tl.sum(centered * centered, axis=0)
    var = tl.div_rn(sum_centered2, N)
    inv = tl.rsqrt(var + 1e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply_rowchunk(X, W, B, S, Y, N: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    offs = chunk * CHUNK + tl.arange(0, CHUNK)
    mask = offs < N
    x_ptrs = X + row * N + offs
    y_ptrs = Y + row * N + offs
    w_ptrs = W + offs
    b_ptrs = B + offs
    mean = tl.load(S + 2 * row)
    inv = tl.load(S + 2 * row + 1)
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    w = tl.load(w_ptrs, mask=mask, other=0.0)
    b = tl.load(b_ptrs, mask=mask, other=0.0)
    y = (x - mean) * inv * w + b
    tl.store(y_ptrs, y, mask=mask)
