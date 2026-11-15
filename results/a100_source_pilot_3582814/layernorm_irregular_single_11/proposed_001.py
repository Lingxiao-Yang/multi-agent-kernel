import triton
import triton.language as tl

@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    count = 0
    mean = tl.zeros([1], dtype=tl.float32)
    m2 = tl.zeros([1], dtype=tl.float32)
    for i in range(0, N, BLOCK):
        idx = i + offset
        mask = idx < N
        x = tl.load(X + row * N + idx, mask, other=0.0)
        for j in range(BLOCK):
            if mask[j]:
                count += 1
                delta = x[j] - mean[0]
                mean[0] += delta / count
                delta2 = x[j] - mean[0]
                m2[0] += delta * delta2
    variance = m2[0] / count if count > 0 else 0.0
    inv = tl.rsqrt(variance + 1e-5)
    tl.store(S + 2 * row, mean[0])
    tl.store(S + 2 * row + 1, inv)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    i = pid * BLOCK + offset
    mask = i < TOTAL
    row = i // N
    col = i % N
    x = tl.load(X + i, mask, other=0.0)
    mean = tl.load(S + 2 * row, mask, other=0.0)
    inv = tl.load(S + 2 * row + 1, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)
    y = (x - mean) * inv * w + b
    tl.store(Y + i, y, mask)
