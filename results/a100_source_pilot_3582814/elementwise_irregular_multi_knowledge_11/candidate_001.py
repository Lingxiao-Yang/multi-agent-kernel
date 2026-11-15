import triton
import triton.language as tl

@triton.jit
def bias_add_relu_fused(X, B, Y, M: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = pid_m * BLOCK
    col_start = pid_n * BLOCK

    offs_m = row_start + tl.arange(0, BLOCK)
    offs_n = col_start + tl.arange(0, BLOCK)

    mask_m = offs_m < M
    mask_n = offs_n < N

    offs_m_broadcast = tl.reshape(offs_m, (BLOCK, 1))
    offs_n_broadcast = tl.reshape(offs_n, (1, BLOCK))

    ptrs = offs_m_broadcast * N + offs_n_broadcast

    # Load X block with mask
    x = tl.load(X + ptrs, mask=(mask_m[:, None] & mask_n[None, :]), other=0.0)

    # Load bias for columns with mask
    b = tl.load(B + offs_n, mask=mask_n, other=0.0)

    # Add bias and apply ReLU
    y = x + b
    y = tl.maximum(y, 0.0)

    # Store result with mask
    tl.store(Y + ptrs, y, mask=(mask_m[:, None] & mask_n[None, :]))
