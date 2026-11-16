import triton
import triton.language as tl

@triton.jit
def layernorm_fused_parallel(X, W, B, Y, N: tl.constexpr, VEC: tl.constexpr):
    row = tl.program_id(0)
    num_chunks = (N + VEC - 1) // VEC
    mean_acc = 0.0
    meansq_acc = 0.0
    for i in range(0, num_chunks):
        chunk_start = i * VEC
        chunk_end = tl.minimum(chunk_start + VEC, N)
        chunk_range = tl.arange(0, VEC)
        chunk_mask = (chunk_range + chunk_start) < N
        chunk_offs = row * N + chunk_start + chunk_range
        x_chunk = tl.load(X + chunk_offs, chunk_mask, other=0.0)
        mean_acc += tl.sum(x_chunk, 0)
        meansq_acc += tl.sum(x_chunk * x_chunk, 0)
    mean = mean_acc / N
    meansq = meansq_acc / N
    var = meansq - mean * mean
    inv = tl.rsqrt(var + 1e-5)
    # Normalize and write output
    for i in range(0, num_chunks):
        chunk_start = i * VEC
        chunk_end = tl.minimum(chunk_start + VEC, N)
        chunk_range = tl.arange(0, VEC)
        chunk_mask = (chunk_range + chunk_start) < N
        chunk_offs = row * N + chunk_start + chunk_range
        x_chunk = tl.load(X + chunk_offs, chunk_mask, other=0.0)
        w_chunk = tl.load(W + chunk_start + chunk_range, chunk_mask, other=0.0)
        b_chunk = tl.load(B + chunk_start + chunk_range, chunk_mask, other=0.0)
        norm_chunk = (x_chunk - mean) * inv * w_chunk + b_chunk
        tl.store(Y + chunk_offs, norm_chunk, chunk_mask)
