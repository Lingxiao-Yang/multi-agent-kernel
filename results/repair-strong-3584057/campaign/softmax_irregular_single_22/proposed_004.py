import triton
import triton.language as tl

@triton.jit
def softmax_fused_blockloop(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # First pass: max reduction over row in BLOCK-sized chunks
    mx = -float('inf')
    offset = 0
    while offset < N:
        idx = offs + offset
        mask = idx < N
        x_ptrs = X + row * N + idx
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        mx = tl.maximum(mx, tl.max(x, 0))
        offset += BLOCK

    # Second pass: exp and sum reduction over row in BLOCK-sized chunks
    denom = 0.0
    offset = 0
    while offset < N:
        idx = offs + offset
        mask = idx < N
        x_ptrs = X + row * N + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_exp = tl.exp(x - mx)
        denom += tl.sum(x_exp, 0)
        offset += BLOCK

    # Third pass: normalization and store
    offset = 0
    while offset < N:
        idx = offs + offset
        mask = idx < N
        x_ptrs = X + row * N + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_exp = tl.exp(x - mx)
        y = x_exp / denom
        y_ptrs = Y + row * N + idx
        tl.store(y_ptrs, y, mask=mask)
        offset += BLOCK
