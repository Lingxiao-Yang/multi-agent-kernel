import triton
import triton.language as tl

@triton.jit
def gelu(X, Y, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    x = tl.load(X+i,i<TOTAL,other=0.)
    tl.store(Y+i,0.5*x*(1.+tl.erf(x*0.7071067811865476)),i<TOTAL)

import torch
def kernel_function(X):
    Y = X.new_empty((4096, 393216))
    gelu[(786432,)](X, Y, 1610612736, 2048, num_warps=4, num_stages=1)
    return Y
