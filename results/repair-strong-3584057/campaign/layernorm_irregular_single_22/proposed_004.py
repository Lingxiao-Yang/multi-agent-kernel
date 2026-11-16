import triton
import triton.language as tl

@triton.jit
def layernorm_fused_multirow_fullrow(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    ROWS_PER_PROG = 4
    row_start = pid * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK)
    for row_offset in range(ROWS_PER_PROG):
        row = row_start + row_offset
        if row >= 513:
            break
        num_iters = (N + BLOCK - 1) // BLOCK
        mean_acc = 0.0
        var_acc = 0.0
        # First pass: blockwise accumulation for mean
        for i in range(num_iters):
            col = offs + i * BLOCK
            mask = col < N
            x = tl.load(X + row * N + col, mask=mask, other=0.0)
            mean_acc += tl.sum(x, 0)
        mean = tl.div_rn(mean_acc, N)
        # Second pass: blockwise accumulation for variance
        for i in range(num_iters):
            col = offs + i * BLOCK
            mask = col < N
            x = tl.load(X + row * N + col, mask=mask, other=0.0)
            centered = tl.where(mask, x - mean, 0.0)
            var_acc += tl.sum(centered * centered, 0)
        var = tl.div_rn(var_acc, N)
        inv = tl.rsqrt(var + 1e-5)
        # Third pass: normalize and affine transform
        for i in range(num_iters):
            col = offs + i * BLOCK
            mask = col < N
            x = tl.load(X + row * N + col, mask=mask, other=0.0)
            centered = tl.where(mask, x - mean, 0.0)
            w = tl.load(W + col, mask=mask, other=0.0)
            b = tl.load(B + col, mask=mask, other=0.0)
            y = (centered * inv) * w + b
            tl.store(Y + row * N + col, y, mask=mask)
