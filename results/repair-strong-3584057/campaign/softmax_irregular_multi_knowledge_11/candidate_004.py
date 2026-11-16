import triton
import triton.language as tl

@triton.jit
def softmax_multi_tile(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    max_acc = -float('inf')
    sum_acc = 0.0
    # First pass: find max value over the row, tiled by BLOCK
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask, other=-float('inf'))
        max_acc = tl.maximum(max_acc, tl.max(x, 0))
    # Second pass: compute sum of exponentials
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask, other=-float('inf'))
        exp_x = tl.exp(x - max_acc)
        sum_acc = sum_acc + tl.sum(exp_x, 0)
    # Third pass: write normalized output tile by tile
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask, other=-float('inf'))
        exp_x = tl.exp(x - max_acc)
        y = exp_x / sum_acc
        tl.store(Y + row * N + cols, y, mask)
