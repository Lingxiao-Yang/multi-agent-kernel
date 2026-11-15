import triton
import triton.language as tl

@triton.jit
def matmul_simt(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), 0, tl.float32)
    for k in range(K):
        a = tl.load(A + rows * K + k, rows < M, other=0).to(tl.float32)
        b = tl.load(B + k * N + cols, cols < N, other=0).to(tl.float32)
        acc += a[:, None] * b[None, :]
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, (rows[:, None] < M) & (cols[None, :] < N))
