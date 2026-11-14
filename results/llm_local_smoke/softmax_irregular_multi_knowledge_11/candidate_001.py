import triton
import triton.language as tl

@triton.jit
def softmax_fused(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + col, col < N, other=-float('inf'))
    mx = tl.max(x, 0)
    exps = tl.exp(x - mx)
    sum_exp = tl.sum(exps, 0)
    tl.store(Y + row * N + col, exps / sum_exp, col < N)
