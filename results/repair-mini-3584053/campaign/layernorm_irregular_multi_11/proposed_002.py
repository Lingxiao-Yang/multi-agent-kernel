import triton
import triton.language as tl

@triton.jit
def layernorm_stats(X, S, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    half = BLOCK // 2
    col_l = tl.arange(0, half)
    col_r = col_l + half
    mask_l = col_l < N
    mask_r = col_r < N

    x_l = tl.load(X + row * N + col_l, mask_l, other=0.0)
    x_r = tl.load(X + row * N + col_r, mask_r, other=0.0)

    sum_l = tl.sum(x_l, 0)
    sum_r = tl.sum(x_r, 0)
    partial_sum = sum_l + sum_r

    # Store partial sums in scratch S temporarily for stage 1
    # Using S as scratch space: store partial sums at 2*row and 2*row+1
    # To enable multi-stage reduction, here we do two-stage partial sums.
    # Stage 1: each thread writes partial sums to S
    tl.store(S + 2 * row, partial_sum, True)

    # Barrier synchronization is implicit as num_stages=2
    tl.next_stage()

    # Stage 2: single thread per row sums the two partial sums (since only one partial per row, just load and process)
    if tl.program_id(0) == 0:
        total_sum = tl.sum(tl.load(S + 2 * tl.arange(0, 513), True, other=0.0))
        mean = tl.div_rn(total_sum, N * 513)
        # Store mean in S for use in second kernel
        tl.store(S, mean, True)

    # Wait for completion
    tl.next_stage()

    # After stage 2, each thread loads mean and computes variance sum
    mean = tl.load(S, True, other=0.0)

    centered_l = tl.where(mask_l, x_l - mean, 0.0)
    centered_r = tl.where(mask_r, x_r - mean, 0.0)

    sum_sq_l = tl.sum(centered_l * centered_l, 0)
    sum_sq_r = tl.sum(centered_r * centered_r, 0)

    sum_sq = sum_sq_l + sum_sq_r
    inv = tl.rsqrt(tl.div_rn(sum_sq, N) + 1e-5)

    tl.store(S + 2 * row, mean, True)
    tl.store(S + 2 * row + 1, inv, True)

@triton.jit
def layernorm_apply(X, W, B, S, Y, TOTAL: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < TOTAL
    row = idx // N
    col = idx % N

    x = tl.load(X + idx, mask, other=0.0)
    mean = tl.load(S + row * 2, mask, other=0.0)
    inv = tl.load(S + row * 2 + 1, mask, other=0.0)
    w = tl.load(W + col, mask, other=0.0)
    b = tl.load(B + col, mask, other=0.0)

    val = (x - mean) * inv * w + b
    tl.store(Y + idx, val, mask)
