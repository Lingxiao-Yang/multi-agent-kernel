import triton
import triton.language as tl

@triton.jit
def matmul_tensor(A, B, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(triton.cdiv(K, BK)):
        ks = start * BK + kk
        a = tl.load(A + rows[:, None] * K + ks[None, :],
                    (rows[:, None] < M) & (ks[None, :] < K), other=0)
        b = tl.load(B + ks[:, None] * N + cols[None, :],
                    (ks[:, None] < K) & (cols[None, :] < N), other=0)
        acc = tl.dot(a, b, acc, input_precision='ieee')
    tl.store(Y + rows[:, None] * N + cols[None, :], acc,
             (rows[:, None] < M) & (cols[None, :] < N))

import torch
def kernel_function(A, B):
    Y = A.new_empty((2048, 4096))
    matmul_tensor[(64, 128)](A, B, Y, 2048, 4096, 8192, 32, 32, 32, num_warps=4, num_stages=1)
    return Y
