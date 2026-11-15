import triton
import triton.language as tl

@triton.jit
def matmul_simt_tiled_shared(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute offsets for rows and columns
    row_offsets = pid_m * BM + tl.arange(0, BM)
    col_offsets = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Software pipeline stages
    for k0 in range(0, K, BK):
        k_offsets = k0 + tl.arange(0, BK)

        # Load tiles of A and B with masking for boundaries
        a_tile = tl.load(A + row_offsets[:, None] * K + k_offsets[None, :], (row_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0).to(tl.float32)
        b_tile = tl.load(B + k_offsets[:, None] * N + col_offsets[None, :], (k_offsets[:, None] < K) & (col_offsets[None, :] < N), other=0).to(tl.float32)

        # Accumulate partial dot products
        acc += tl.dot(a_tile, b_tile)

    # Store the result with boundary masking
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(Y + row_offsets[:, None] * N + col_offsets[None, :], acc, mask)
