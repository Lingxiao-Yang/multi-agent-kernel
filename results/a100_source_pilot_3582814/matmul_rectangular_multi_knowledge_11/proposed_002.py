import triton
import triton.language as tl

@triton.jit
def matmul_shared_pipeline_double_buffered(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_offsets = pid_m * BM + tl.arange(0, BM)
    col_offsets = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    # Allocate double-buffered shared memory for tiles of A and B
    sA = tl.shared_memory((2, BM, BK), dtype=tl.float16)
    sB = tl.shared_memory((2, BK, BN), dtype=tl.float16)

    # Initialize double buffer index
    buf_idx = 0

    # Preload first tile from global memory to shared memory
    k0 = 0
    k_offsets = k0 + tl.arange(0, BK)
    a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
    b_mask = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)

    a_tile = tl.load(A + row_offsets[:, None] * K + k_offsets[None, :], a_mask, other=0)
    b_tile = tl.load(B + k_offsets[:, None] * N + col_offsets[None, :], b_mask, other=0)

    sA[buf_idx, :, :] = a_tile
    sB[buf_idx, :, :] = b_tile

    tl.barrier()

    for k0 in range(BK, K + BK, BK):
        next_buf_idx = 1 - buf_idx

        # Asynchronously preload the next tile if within bounds
        if k0 < K:
            k_offsets = k0 + tl.arange(0, BK)
            a_mask = (row_offsets[:, None] < M) & (k_offsets[None, :] < K)
            b_mask = (k_offsets[:, None] < K) & (col_offsets[None, :] < N)

            a_tile = tl.load(A + row_offsets[:, None] * K + k_offsets[None, :], a_mask, other=0)
            b_tile = tl.load(B + k_offsets[:, None] * N + col_offsets[None, :], b_mask, other=0)

            sA[next_buf_idx, :, :] = a_tile
            sB[next_buf_idx, :, :] = b_tile

        tl.barrier()

        # Compute on the current shared memory buffers
        a_shared = sA[buf_idx, :, :].to(tl.float32)
        b_shared = sB[buf_idx, :, :].to(tl.float32)

        acc += tl.dot(a_shared, b_shared)

        buf_idx = next_buf_idx

    # Write the result to global memory with boundary masking
    mask = (row_offsets[:, None] < M) & (col_offsets[None, :] < N)
    tl.store(Y + row_offsets[:, None] * N + col_offsets[None, :], acc, mask)
