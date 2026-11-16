import triton
import triton.language as tl

@triton.jit
def softmax_fused_blocked_v2(X, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, RB)
    idx = row * N + offs
    mask = offs < N
    # Load input row with padding masked
    x = tl.load(X + idx, mask=mask, other=-float('inf'))
    # Compute row max (ignoring padding)
    mx = tl.max(x, 0)
    x_centered = x - mx
    # Mask again so padding does not contribute to exp/sum
    exp_x = tl.exp(x_centered)
    exp_x = exp_x * mask
    denom = tl.sum(exp_x, 0)
    # Avoid division by zero (should not occur if N > 0)
    denom = tl.where(denom == 0, 1, denom)
    res = exp_x / denom
    tl.store(Y + idx, res, mask=mask)
