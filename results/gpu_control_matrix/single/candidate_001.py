"""Readable starting kernels and optional transformations; no autotuning.

The baseline uses ordinary SIMT matmul, separate reduction/application
passes for softmax/layernorm, and two elementwise launches. All allocation,
launching, references and timing live outside candidate-editable code.
"""
import triton
import triton.language as tl


@triton.jit
def matmul_simt(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), 0, tl.float32)
    for k in range(K):
        a = tl.load(A + rows * K + k, rows < M, other=0).to(tl.float32)
        b = tl.load(B + k * N + cols, cols < N, other=0).to(tl.float32)
        acc += a[:, None] * b[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def matmul_tensor(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(K, BK)):
        ks = start * BK + kk
        a = tl.load(A + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B + ks[:, None] * N + cols[None, :],
                    (ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))


@triton.jit
def softmax_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    mx = tl.max(x, 0)
    denom = tl.sum(tl.exp(x - mx), 0)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, denom)


@triton.jit
def softmax_apply(X, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    x = tl.load(X + i, i < TOTAL, other=0)
    mx = tl.load(S + row * 2, i < TOTAL, other=0)
    denom = tl.load(S + row * 2 + 1, i < TOTAL, other=1)
    tl.store(Y + i, tl.exp(x - mx) / denom, i < TOTAL)


@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    e = tl.exp(x - tl.max(x, 0))
    tl.store(Y + row * N + col, e / tl.sum(e, 0), col < N)


@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0)
    mean = tl.div_rn(tl.sum(x, 0), N)
    centered = tl.where(col < N, x - mean, 0)
    inv = tl.rsqrt(tl.sum(centered * centered, 0) / N + 1.e-5)
    tl.store(S + 2 * row, mean)
    tl.store(S + 2 * row + 1, inv)


@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // N
    col = i % N
    x = tl.load(X + i, i < TOTAL, other=0)
    mean = tl.load(S + row * 2, i < TOTAL, other=0)
    inv = tl.load(S + row * 2 + 1, i < TOTAL, other=0)
    w = tl.load(W + col, i < TOTAL, other=0)
    b = tl.load(B + col, i < TOTAL, other=0)
    tl.store(Y + i, (x - mean) * inv * w + b, i < TOTAL)


@triton.jit
def layernorm_fused(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=0)
    mean = tl.div_rn(tl.sum(x, 0), N)
    centered = tl.where(col < N, x - mean, 0)
    inv = tl.rsqrt(tl.sum(centered * centered, 0) / N + 1.e-5)
    w = tl.load(W + col, col < N, other=0)
    b = tl.load(B + col, col < N, other=0)
    tl.store(Y + row * N + col, centered * inv * w + b, col < N)


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


@triton.jit
def bias_relu(X, B, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < TOTAL, other=0)
    b = tl.load(B + i % N, i < TOTAL, other=0)
    tl.store(Y + i, tl.maximum(x + b, 0), i < TOTAL)
