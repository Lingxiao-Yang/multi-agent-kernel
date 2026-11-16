import triton
import triton.language as tl

@triton.jit
def softmax_blockloop_unroll2_reduced_regs(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    # Compute max in block-looped, register-sparse fashion
    max_val = -float('inf')
    for base in range(0, ((N + BLOCK - 1) // BLOCK) * BLOCK, BLOCK):
        idx = base + offs
        mask = idx < N
        x_ptrs = X + row * N + idx
        x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
        # Reduce across only valid elements
        max_val = tl.maximum(max_val, tl.max(x, 0))
    # Compute sum(exp(x - max)) block-looped
    denom = 0.0
    for base in range(0, ((N + BLOCK - 1) // BLOCK) * BLOCK, BLOCK):
        idx = base + offs
        mask = idx < N
        x_ptrs = X + row * N + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        e = tl.exp(x - max_val)
        denom = denom + tl.sum(e, 0)
    # Write output in block-looped steps
    for base in range(0, ((N + BLOCK - 1) // BLOCK) * BLOCK, BLOCK):
        idx = base + offs
        mask = idx < N
        x_ptrs = X + row * N + idx
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        e = tl.exp(x - max_val) / denom
        y_ptrs = Y + row * N + idx
        tl.store(y_ptrs, e, mask=mask)
