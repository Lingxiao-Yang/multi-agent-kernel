import triton
import triton.language as tl

@triton.jit
def softmax_exp_sum(X, S, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, RB)
    mask = col < N

    # Load input row with masking padded columns
    x = tl.load(X + row * N + col, mask, other=-float('inf'))

    # Compute max over valid elements
    mx = tl.max(x, 0)
    # Store max in compact layout: 2 floats per row
    tl.store(S + row * 2, mx)

    # Compute sum of exp(x - max) over valid elements
    exp_x = tl.exp(x - mx)
    exp_sum = tl.sum(exp_x, 0)
    # Store sum in compact layout
    tl.store(S + row * 2 + 1, exp_sum)

@triton.jit
def softmax_sum_reduce(S, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    # Load max and sum from compact layout
    mx = tl.load(S + row * 2)
    exp_sum = tl.load(S + row * 2 + 1)
    # Store them back unchanged (placeholder for multi-stage reductions if needed)
    tl.store(S + row * 2, mx)
    tl.store(S + row * 2 + 1, exp_sum)

@triton.jit
def softmax_apply(X, S, Y, N: tl.constexpr, RB: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, RB)
    mask = col < N

    # Load input with masking for padded elements
    x = tl.load(X + row * N + col, mask, other=0.0)
    # Load scalar max and denom without mask (scalar pointers)
    mx = tl.load(S + row * 2)
    denom = tl.load(S + row * 2 + 1)

    # Compute normalized softmax value per element
    val = tl.exp(x - mx) / denom
    # Store with masking
    tl.store(Y + row * N + col, val, mask)
