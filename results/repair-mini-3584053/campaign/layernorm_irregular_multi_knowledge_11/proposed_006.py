import triton
import triton.language as tl

@triton.jit
def layernorm_fused_pipelined_repair(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)  # constexpr

    # --- Mean accumulation with explicit double buffering and masking ---
    mean_acc = 0.0
    count = 0.0

    # Load first block
    block_start = 0
    col = block_start + offs
    mask = col < N
    x_curr = tl.load(X + row * N + col, mask, other=0.0)

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

    # --- Variance accumulation with explicit double buffering and masking ---
    var_acc = 0.0

    # Load first block
    block_start = 0
    col = block_start + offs
    mask = col < N
    x_curr = tl.load(X + row * N + col, mask, other=0.0)

    for block_start_next in range(BLOCK, N + BLOCK, BLOCK):
        col_next = block_start_next + offs
        mask_next = col_next < N
        x_next = tl.zeros((BLOCK,), dtype=tl.float32)
        if block_start_next < N:
            x_next = tl.load(X + row * N + col_next, mask_next, other=0.0)

        centered = tl.where(mask, x_curr - mean, 0.0)
        var_acc += tl.sum(centered * centered, 0)

        x_curr = x_next
        mask = mask_next

    var = var_acc / count
    inv_std = tl.rsqrt(var + 1e-5)

    # --- Normalization and store with explicit double buffering and masking ---
    block_start = 0
    col = block_start + offs
    mask = col < N
    x_curr = tl.load(X + row * N + col, mask, other=0.0)
    w_curr = tl.load(W + col, mask, other=0.0)
    b_curr = tl.load(B + col, mask, other=0.0)

    for block_start_next in range(BLOCK, N + BLOCK, BLOCK):
        col_next = block_start_next + offs
        mask_next = col_next < N
        x_next = tl.zeros((BLOCK,), dtype=tl.float32)
        w_next = tl.zeros((BLOCK,), dtype=tl.float32)
        b_next = tl.zeros((BLOCK,), dtype=tl.float32)
        if block_start_next < N:
            x_next = tl.load(X + row * N + col_next, mask_next, other=0.0)
            w_next = tl.load(W + col_next, mask_next, other=0.0)
            b_next = tl.load(B + col_next, mask_next, other=0.0)

        y = (x_curr - mean) * inv_std * w_curr + b_curr
        tl.store(Y + row * N + col, y, mask)

        x_curr = x_next
        w_curr = w_next
        b_curr = b_next
        mask = mask_next

    # Store last block
    y = (x_curr - mean) * inv_std * w_curr + b_curr
    tl.store(Y + row * N + col, y, mask)
