import triton
import triton.language as tl
import torch

@triton.jit
def _matvec_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=ACC_DTYPE)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k * stride_bk

        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0)
        b = tl.load(b_ptrs, mask=k_mask, other=0)
        acc += tl.sum(a.to(ACC_DTYPE) * b[None, :].to(ACC_DTYPE), axis=1)

    out = acc.to(OUT_DTYPE)
    c_ptrs = c_ptr + offs_m * stride_cm
    tl.store(c_ptrs, out, mask=m_mask)

def kernel_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if not A.is_cuda or not B.is_cuda:
        raise ValueError("A and B must be CUDA tensors.")
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError("A and B must be torch.bfloat16 tensors.")
    if A.ndim != 2:
        raise ValueError("A must be 2D [M, K].")
    M, K = A.shape
    if B.ndim == 1:
        if B.shape[0] != K:
            raise ValueError(f"When B is 1D, expected shape [K]={K}, but got {tuple(B.shape)}")
        Bv = B.view(K, 1)
    elif B.ndim == 2:
        if B.shape[0] != K or B.shape[1] != 1:
            raise ValueError(f"B must be [K, 1], got {tuple(B.shape)} (K must match A.shape[1])")
        Bv = B
    else:
        raise ValueError("B must be 1D [K] or 2D [K, 1].")

    C = torch.empty((M, 1), device=A.device, dtype=A.dtype)
    stride_am, stride_ak = A.stride()
    stride_bk, stride_bn = Bv.stride()
    stride_cm, stride_cn = C.stride()

    BLOCK_M = 64
    BLOCK_K = 64
    ACC_DTYPE = tl.float32
    OUT_DTYPE = tl.bfloat16

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]),)

    _matvec_kernel[grid](
        A,
        Bv,
        C,
        M,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        ACC_DTYPE=ACC_DTYPE,
        OUT_DTYPE=OUT_DTYPE,
        num_warps=8,
        num_stages=3,
    )

    return C