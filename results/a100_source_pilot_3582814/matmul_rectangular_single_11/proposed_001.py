import triton
import triton.language as tl

@triton.jit
def matmul_simt(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Double buffering along K dimension
    for k_start in range(0, K, 2*BK):
        # Load first BK block
        k0 = k_start + tl.arange(0, BK)
        mask_a0 = (rows[:, None] < M) & (k0[None, :] < K)
        mask_b0 = (k0[:, None] < K) & (cols[None, :] < N)
        a0 = tl.load(A + rows[:, None]*K + k0[None, :], mask=mask_a0, other=0).to(tl.float32)
        b0 = tl.load(B + k0[:, None]*N + cols[None, :], mask=mask_b0, other=0).to(tl.float32)

        # Load second BK block if within bounds
        k1_start = k_start + BK
        if k1_start < K:
            k1 = k1_start + tl.arange(0, BK)
            mask_a1 = (rows[:, None] < M) & (k1[None, :] < K)
            mask_b1 = (k1[:, None] < K) & (cols[None, :] < N)
            a1 = tl.load(A + rows[:, None]*K + k1[None, :], mask=mask_a1, other=0).to(tl.float32)
            b1 = tl.load(B + k1[:, None]*N + cols[None, :], mask=mask_b1, other=0).to(tl.float32)
        else:
            a1 = tl.zeros((BM, BK), dtype=tl.float32)
            b1 = tl.zeros((BK, BN), dtype=tl.float32)

        acc += tl.dot(a0, b0)
        acc += tl.dot(a1, b1)

    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(Y + rows[:, None]*N + cols[None, :], acc, mask)
