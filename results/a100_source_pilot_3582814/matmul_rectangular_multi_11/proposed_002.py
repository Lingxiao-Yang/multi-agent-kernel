import triton
import triton.language as tl

@triton.jit
def matmul_shared_tiled_vec_v2(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Allocate shared memory for tiles of A and B
    smem_a = tl.shared_memory(BM * BK, dtype=tl.float16)
    smem_b = tl.shared_memory(BK * BN, dtype=tl.float16)

    # Vector length for loading/storing
    vec_len = 4

    for k_block_start in range(0, K, BK):
        k_offs = k_block_start + tl.arange(0, BK)

        # Load tile of A: shape [BM, BK]
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a_ptrs = A + offs_m[:, None] * K + k_offs[None, :]
        a_tile = tl.load(a_ptrs, mask_a, other=0).to(tl.float32)

        # Load tile of B: shape [BK, BN]
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b_ptrs = B + k_offs[:, None] * N + offs_n[None, :]
        b_tile = tl.load(b_ptrs, mask_b, other=0).to(tl.float32)

        # Store tiles to shared memory with vectorized stores
        # Flatten indices for shared memory
        offs_a = tl.arange(0, BM * BK)
        a_row = offs_a // BK
        a_col = offs_a % BK
        mask_smem_a = (offs_m[a_row] < M) & (k_block_start + a_col < K)
        tl.store(smem_a + offs_a, a_tile.to(tl.float16).reshape(-1), mask_smem_a)

        offs_b = tl.arange(0, BK * BN)
        b_row = offs_b // BN
        b_col = offs_b % BN
        mask_smem_b = (k_block_start + b_row < K) & (offs_n[b_col] < N)
        tl.store(smem_b + offs_b, b_tile.to(tl.float16).reshape(-1), mask_smem_b)

        # Synchronize to ensure shared memory tiles are ready
        tl.barrier(0)

        # Compute acc += dot(A_tile, B_tile) using shared memory
        # Load shared memory tiles as float32
        a_sh = tl.load(smem_a + a_row[:, None] * BK + a_col[None, :], mask_smem_a[:, None], other=0).to(tl.float32)
        b_sh = tl.load(smem_b + b_row[:, None] * BN + b_col[None, :], mask_smem_b[:, None], other=0).to(tl.float32)

        acc += tl.dot(a_sh, b_sh)

        # Synchronize before next k_block
        tl.barrier(0)

    # Store output tile
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask)
