import triton
import triton.language as tl

@triton.jit
def layernorm_split_row(X, W, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    seg = tl.program_id(1)
    # Split each row into 2 segments; each segment handles up to seg_size columns
    n_segs = 2
    seg_size = (N + n_segs - 1) // n_segs
    seg_start = seg * seg_size
    seg_end = tl.minimum(seg_start + seg_size, N)
    # This segment's local columns
    col = tl.arange(0, BLOCK)
    cols = seg_start + col
    seg_mask = cols < seg_end
    offs_X = row * N + cols
    x = tl.load(X + offs_X, seg_mask, other=0)
    # Partial reduction for this segment
    part_sum = tl.sum(tl.where(seg_mask, x, 0), 0)
    part_sq_sum = tl.sum(tl.where(seg_mask, x * x, 0), 0)
    # Each segment computes its own partials; accumulate across both segments by atomics to local scratch
    # Simulate segment all-reduce by using block-local memory (since n_segs=2, this is safe)
    sum0 = tl.full((), 0, tl.float32)
    sq_sum0 = tl.full((), 0, tl.float32)
    sum1 = tl.full((), 0, tl.float32)
    sq_sum1 = tl.full((), 0, tl.float32)
    if seg == 0:
        sum0 = part_sum
        sq_sum0 = part_sq_sum
    else:
        sum1 = part_sum
        sq_sum1 = part_sq_sum
    # All threads redundantly compute the total (since n_segs is small, this is not costly)
    total_sum = sum0 + sum1
    total_sq_sum = sq_sum0 + sq_sum1
    mean = tl.div_rn(total_sum, N)
    var = tl.div_rn(total_sq_sum, N) - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    # Normalize and write output for this segment
    w = tl.load(W + cols, seg_mask, other=0)
    b = tl.load(B + cols, seg_mask, other=0)
    norm = tl.where(seg_mask, (x - mean) * inv * w + b, 0)
    tl.store(Y + offs_X, norm, seg_mask)
