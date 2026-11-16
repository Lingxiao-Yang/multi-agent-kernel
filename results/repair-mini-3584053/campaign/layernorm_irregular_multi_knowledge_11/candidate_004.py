import triton
import triton.language as tl

@triton.jit
def layernorm_fused_pipelined(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mean_acc = 0.0
    var_acc = 0.0
    count = 0.0

    offs = tl.arange(0, BLOCK)  # constexpr

    # --- Accumulate mean with software pipelining ---
    # Initialize pipeline
    block_start = 0
    col = block_start + offs
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0.0)

    for block_start_next in range(BLOCK, N + BLOCK, BLOCK):
        col_next = block_start_next + offs
        mask_next = col_next < N
        if block_start_next < N:
            x_next = tl.load(X + row * N + col_next, mask_next, other=0.0)
        else:
            x_next = tl.zeros((BLOCK,), dtype=tl.float32)

        mean_acc += tl.sum(tl.where(mask, x, 0.0), 0)
        count += tl.sum(tl.where(mask, 1.0, 0.0), 0)

        x = x_next
        mask = mask_next

    mean = mean_acc / count

    # --- Accumulate variance with software pipelining ---
    block_start = 0
    col = block_start + offs
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0.0)

    for block_start_next in range(BLOCK, N + BLOCK, BLOCK):
        col_next = block_start_next + offs
        mask_next = col_next < N
        if block_start_next < N:
            x_next = tl.load(X + row * N + col_next, mask_next, other=0.0)
        else:
            x_next = tl.zeros((BLOCK,), dtype=tl.float32)

        centered = tl.where(mask, x - mean, 0.0)
        var_acc += tl.sum(centered * centered, 0)

        x = x_next
        mask = mask_next

    var = var_acc / count
    inv_std = tl.rsqrt(var + 1e-5)

    # --- Normalize and store output with software pipelining ---
    block_start = 0
    col = block_start + offs
    mask = col < N
    x = tl.load(X + row * N + col, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)

    for block_start_next in range(BLOCK, N + BLOCK, BLOCK):
        col_next = block_start_next + offs
        mask_next = col_next < N
        if block_start_next < N:
            x_next = tl.load(X + row * N + col_next, mask_next, other=0.0)
            w_next = tl.load(W + col_next, mask_next, other=0.0)
            b_next = tl.load(B + col_next, mask_next, other=0.0)
        else:
            x_next = tl.zeros((BLOCK,), dtype=tl.float32)
            w_next = tl.zeros((BLOCK,), dtype=tl.float32)
            b_next = tl.zeros((BLOCK,), dtype=tl.float32)

        y = (x - mean) * inv_std * w + b
        tl.store(Y + row * N + col, y, mask)

        x = x_next
        w = w_next
        b = b_next
        mask = mask_next

    # Store last block
    y = (x - mean) * inv_std * w + b
    tl.store(Y + row * N + col, y, mask)

