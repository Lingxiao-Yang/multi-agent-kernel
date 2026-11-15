import triton
import triton.language as tl

@triton.jit
def matmul_simt_shared(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Allocate shared memory tiles
    shared_A = tl.shared_memory((BM, BK), tl.float16)
    shared_B = tl.shared_memory((BK, BN), tl.float16)

    for k_start in range(0, K, BK):
        k_offs = k_start + tl.arange(0, BK)

        # Load tiles from global memory to shared memory with masking
        mask_A = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_B = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        # Double buffering: use ping-pong buffer index
        # We use a single buffer here, but insert barrier before and after
        # Load A tile
        for i in range(BM):
            for j in range(BK):
                idx_m = offs_m[i]
                idx_k = k_offs[j]
                val = tl.load(A + idx_m * K + idx_k, mask=idx_m < M and idx_k < K, other=0).to(tl.float32)
                shared_A[i, j] = val.to(tl.float16)

        # Load B tile
        for i in range(BK):
            for j in range(BN):
                idx_k = k_offs[i]
                idx_n = offs_n[j]
                val = tl.load(B + idx_k * N + idx_n, mask=idx_k < K and idx_n < N, other=0).to(tl.float32)
                shared_B[i, j] = val.to(tl.float16)

        tl.barrier()

        # Compute on shared memory tiles
        for k_inner in range(BK):
            a = shared_A[:, k_inner].to(tl.float32)
            b = shared_B[k_inner, :].to(tl.float32)
            acc += a[:, None] * b[None, :]

        tl.barrier()

    # Store the result
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask)