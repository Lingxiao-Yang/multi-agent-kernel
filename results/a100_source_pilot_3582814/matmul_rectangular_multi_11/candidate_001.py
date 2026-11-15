import triton
import triton.language as tl

@triton.jit
def matmul_simt_tiled(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    for k_block_start in range(0, K, BK):
        k_offs = k_block_start + tl.arange(0, BK)

        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A + offs_m[:, None] * K + k_offs[None, :], a_mask, 0).to(tl.float32)
        b = tl.load(B + k_offs[:, None] * N + offs_n[None, :], b_mask, 0).to(tl.float32)

        acc += tl.dot(a, b)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask)
