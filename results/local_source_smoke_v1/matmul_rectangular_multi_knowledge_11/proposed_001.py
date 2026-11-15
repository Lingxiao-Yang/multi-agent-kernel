import triton
import triton.language as tl

@triton.jit
def matmul_tiled_pipeline(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Software pipeline buffers
    a_ptrs = tl.zeros(BK, dtype=tl.int32)
    b_ptrs = tl.zeros(BK, dtype=tl.int32)

    # Initialize pipeline: load first tiles
    k = 0
    offs_k = k + tl.arange(0, BK)
    mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    a_tile = tl.load(A + offs_m[:, None] * K + offs_k[None, :], mask=mask_a, other=0).to(tl.float32)
    b_tile = tl.load(B + offs_k[:, None] * N + offs_n[None, :], mask=mask_b, other=0).to(tl.float32)

    # Pipeline loop with 3 stages
    for k in range(BK, K, BK):
        # Preload next tiles
        offs_k_next = k + tl.arange(0, BK)
        mask_a_next = (offs_m[:, None] < M) & (offs_k_next[None, :] < K)
        mask_b_next = (offs_k_next[:, None] < K) & (offs_n[None, :] < N)
        a_tile_next = tl.load(A + offs_m[:, None] * K + offs_k_next[None, :], mask=mask_a_next, other=0).to(tl.float32)
        b_tile_next = tl.load(B + offs_k_next[:, None] * N + offs_n[None, :], mask=mask_b_next, other=0).to(tl.float32)

        # Compute dot product for current tiles
        acc += tl.dot(a_tile, b_tile)

        # Advance pipeline
        a_tile = a_tile_next
        b_tile = b_tile_next

    # Last accumulation for the final tile
    acc += tl.dot(a_tile, b_tile)

    # Store result with boundary masking
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask)
