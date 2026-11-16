import triton
import triton.language as tl

@triton.jit
def softmax_fused_blockloop_unrolled(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # Unroll factor (process two BLOCKs per iteration)
    UNROLL = 2
    mx = -float('inf')
    offset = 0
    while offset < N:
        for u in range(UNROLL):
            idx = offs + offset + u * BLOCK
            mask = idx < N
            x_ptrs = X + row * N + idx
            x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
            mx = tl.maximum(mx, tl.max(x, 0))
        offset += UNROLL * BLOCK

    denom = 0.0
    offset = 0
    while offset < N:
        for u in range(UNROLL):
            idx = offs + offset + u * BLOCK
            mask = idx < N
            x_ptrs = X + row * N + idx
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            x_exp = tl.exp(x - mx)
            denom += tl.sum(x_exp, 0)
        offset += UNROLL * BLOCK

    offset = 0
    while offset < N:
        for u in range(UNROLL):
            idx = offs + offset + u * BLOCK
            mask = idx < N
            x_ptrs = X + row * N + idx
            x = tl.load(x_ptrs, mask=mask, other=0.0)
            x_exp = tl.exp(x - mx)
            y = x_exp / denom
            y_ptrs = Y + row * N + idx
            tl.store(y_ptrs, y, mask=mask)
        offset += UNROLL * BLOCK
