import triton
import triton.language as tl

@triton.jit
def softmax_pipelined_split(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    half = 512  # constexpr half of BLOCK=1024
    offs_lo = tl.arange(0, half)
    offs_hi = tl.arange(512, BLOCK)
    mask_lo = offs_lo < N
    mask_hi = offs_hi < N

    # Stage 1: Load halves and compute partial max reductions
    x_lo = tl.load(X + row * N + offs_lo, mask_lo, other=-float('inf'))
    x_hi = tl.load(X + row * N + offs_hi, mask_hi, other=-float('inf'))
    max_lo = tl.max(x_lo, axis=0)
    max_hi = tl.max(x_hi, axis=0)
    mx = tl.maximum(max_lo, max_hi)

    # Stage 2: Compute exponentials and partial sums, then sum denom
    exps_lo = tl.where(mask_lo, tl.exp(x_lo - mx), 0.0)
    exps_hi = tl.where(mask_hi, tl.exp(x_hi - mx), 0.0)
    denom_lo = tl.sum(exps_lo, axis=0)
    denom_hi = tl.sum(exps_hi, axis=0)
    denom = denom_lo + denom_hi

    # Normalize and write back the output in two halves with masking
    y_lo = tl.where(mask_lo, exps_lo / denom, 0.0)
    y_hi = tl.where(mask_hi, exps_hi / denom, 0.0)
    tl.store(Y + row * N + offs_lo, y_lo, mask_lo)
    tl.store(Y + row * N + offs_hi, y_hi, mask_hi)
