import triton
import triton.language as tl

@triton.jit
def bias_add_relu_fused(X, B, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    pid_m = pid // 49
    pid_n = pid % 49

    row_start = pid_m * BLOCK
    col_start = pid_n * BLOCK

    offs_m = row_start + tl.arange(0, BLOCK)
    offs_n = col_start + tl.arange(0, BLOCK)

    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_m_broadcast = tl.reshape(offs_m, (BLOCK, 1))
    offs_n_broadcast = tl.reshape(offs_n, (1, BLOCK))

    ptrs = offs_m_broadcast * N + offs_n_broadcast

    mask = mask_m[:, None] & mask_n[None, :]

    x = tl.load(X + ptrs, mask=mask, other=0.0)
    b = tl.load(B + offs_n, mask=mask_n, other=0.0)

    y = x + b
    y = tl.maximum(y, 0.0)

    tl.store(Y + ptrs, y, mask=mask)
