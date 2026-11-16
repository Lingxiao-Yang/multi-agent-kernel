import triton
import triton.language as tl

@triton.jit
def softmax_parallel_chunks(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    CHUNK_SIZE = BLOCK // 2
    cols = tl.arange(0, CHUNK_SIZE) + chunk * CHUNK_SIZE
    offs = row * N + cols
    mask = cols < N
    # Load input row chunk with padding masked
    x = tl.load(X + offs, mask, other=-float('inf'))
    # Each chunk computes local max
    local_mx = tl.max(x, 0)
    # Reduce max across chunks using shared mem
    mx = tl.max_contiguous(local_mx, 0)
    mx = tl.broadcast_to(mx, x.shape)
    # Center and exponentiate
    x_centered = x - mx
    exp_x = tl.exp(x_centered)
    # Local sum for denominator
    local_sum = tl.sum(exp_x, 0)
    # Reduce denominator across chunks
    denom = tl.sum(local_sum, 0)
    denom = tl.broadcast_to(denom, x.shape)
    # Normalize
    y = exp_x / denom
    # Store results to output, mask padding
    tl.store(Y + offs, y, mask)
