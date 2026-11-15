import triton
import triton.language as tl

@triton.jit
def matmul_simt(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), tl.float32)

    # We use a double buffer pipeline with 2 stages
    k_ptrs = [0, BK]
    # Load first tiles
    k = 0
    k_range = k + tl.arange(0, BK)
    mask_a = (rows[:, None] < M) & (k_range[None, :] < K)
    a = tl.load(A + rows[:, None] * K + k_range[None, :], mask=mask_a, other=0).to(tl.float32)
    mask_b = (k_range[:, None] < K) & (cols[None, :] < N)
    b = tl.load(B + k_range[:, None] * N + cols[None, :], mask=mask_b, other=0).to(tl.float32)

    for i in range(k + BK, K + BK, BK):
        # Next k range
        k_next = i
        k_range_next = k_next + tl.arange(0, BK)
        mask_a_next = (rows[:, None] < M) & (k_range_next[None, :] < K)
        mask_b_next = (k_range_next[:, None] < K) & (cols[None, :] < N)

        # Prefetch next tiles
        a_next = tl.load(A + rows[:, None] * K + k_range_next[None, :], mask=mask_a_next, other=0).to(tl.float32)
        b_next = tl.load(B + k_range_next[:, None] * N + cols[None, :], mask=mask_b_next, other=0).to(tl.float32)

        # Compute dot product on current tiles
        acc += tl.dot(a, b)

        # Swap buffers
        a = a_next
        b = b_next

    # Last dot product
    acc += tl.dot(a, b)

    mask = (rows[:, None] < M) & (cols[None, :] < N)
    tl.store(Y + rows[:, None] * N + cols[None, :], acc, mask)
