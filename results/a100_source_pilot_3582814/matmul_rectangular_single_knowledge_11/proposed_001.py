import triton
import triton.language as tl

@triton.jit
def matmul_simt(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Software pipeline with 2 stages to overlap loads and compute
    for k in range(0, K, BK):
        offs_k = k + tl.arange(0, BK)
        # Load tiles with boundary mask
        a = tl.load(A + offs_m[:, None] * K + offs_k[None, :], (offs_m[:, None] < M) & (offs_k[None, :] < K), other=0).to(tl.float32)
        b = tl.load(B + offs_k[:, None] * N + offs_n[None, :], (offs_k[:, None] < K) & (offs_n[None, :] < N), other=0).to(tl.float32)
        acc += tl.dot(a, b)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask)
