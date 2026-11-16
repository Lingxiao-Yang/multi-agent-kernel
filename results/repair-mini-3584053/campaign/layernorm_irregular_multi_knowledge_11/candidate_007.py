import triton
import triton.language as tl

@triton.jit
def layernorm_fused_pipelined_meanonly(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # constexpr

    # --- Pipeline mean accumulation only with explicit double buffering ---
    mean_acc = 0.0
    count = 0.0

    # Load first block
    block_start = 0
    col = block_start + offs
    mask = col < N
    x_curr = tl.load(X + row * N + col, mask, other=0.0)

    # Iterate over blocks with double buffering
    for block_start_next in range(BLOCK, N + BLOCK, BLOCK):
        col_next = block_start_next + offs
        mask_next = col_next < N
        x_next = tl.zeros((BLOCK,), dtype=tl.float32)
        if block_start_next < N:
            x_next = tl.load(X + row * N + col_next, mask_next, other=0.0)

        # Accumulate from current block
        mean_acc += tl.sum(tl.where(mask, x_curr, 0.0), 0)
        count += tl.sum(tl.where(mask, 1.0, 0.0), 0)

        # Advance pipeline
        x_curr = x_next
        mask = mask_next

    mean = mean_acc / count

    # --- Variance accumulation (non-pipelined) ---
    var_acc = 0.0
    for block_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = block_start + offs
        mask = col < N
        x = tl.load(X + row * N + col, mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        var_acc += tl.sum(centered * centered, 0)

    var = var_acc / count
    inv_std = tl.rsqrt(var + 1e-5)

    # --- Normalization and store (non-pipelined) ---
    for block_start in range(0, N, BLOCK):
        offs = tl.arange(0, BLOCK)
        col = block_start + offs
        mask = col < N
        x = tl.load(X + row * N + col, mask, other=0.0)
        w = tl.load(W + col, mask, other=0.0)
        b = tl.load(B + col, mask, other=0.0)
        y = (x - mean) * inv_std * w + b
        tl.store(Y + row * N + col, y, mask)
