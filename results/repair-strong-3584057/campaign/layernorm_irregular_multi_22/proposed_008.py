import triton
import triton.language as tl

# Pass 1: compute mean for each row, store to T[row]
@triton.jit
def mean_rowwise_pass(X, T, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    acc = 0.0
    n_tiles = (N + BLOCK - 1) // BLOCK
    for tile in range(n_tiles):
        start = tile * BLOCK
        c = cols + start
        mask = c < N
        x_ptrs = X + row * N + c
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(x, 0)
    mean = tl.div_rn(acc, N)
    tl.store(T + row, mean)

# Pass 2: compute variance (wrt row mean), normalize, affine, store Y
@triton.jit
def varnorm_rowwise_pass(X, W, B, T, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    mean = tl.load(T + row)
    cols = tl.arange(0, BLOCK)
    acc = 0.0
    n_tiles = (N + BLOCK - 1) // BLOCK
    # Compute variance (sum of squared deviations from mean)
    for tile in range(n_tiles):
        start = tile * BLOCK
        c = cols + start
        mask = c < N
        x_ptrs = X + row * N + c
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        d = tl.where(mask, x - mean, 0.0)
        acc += tl.sum(d * d, 0)
    var = tl.div_rn(acc, N)
    inv_std = tl.rsqrt(var + 1e-5)
    # Normalize and affine-transform
    for tile in range(n_tiles):
        start = tile * BLOCK
        c = cols + start
        mask = c < N
        x_ptrs = X + row * N + c
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        centered = tl.where(mask, x - mean, 0.0)
        w = tl.load(W + c, mask=mask, other=0.0)
        b = tl.load(B + c, mask=mask, other=0.0)
        normed = (centered * inv_std) * w + b
        tl.store(Y + row * N + c, normed, mask=mask)
