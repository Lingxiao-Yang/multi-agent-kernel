import triton
import triton.language as tl

@triton.jit
def rms_norm(X, Y, C: tl.constexpr, SPATIAL: tl.constexpr, BLOCK: tl.constexpr):
    batch = tl.program_id(1)
    feature = tl.arange(0,C)
    position = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    offset = (batch*C + feature[:,None])*SPATIAL + position[None,:]
    x = tl.load(X+offset,position[None,:]<SPATIAL,other=0.)
    scale = tl.rsqrt(tl.sum(x*x,0)/C + 1e-5)
    tl.store(Y+offset,x*scale[None,:],position[None,:]<SPATIAL)

import torch
def kernel_function(X):
    Y = X.new_empty((112, 64, 512, 512))
    rms_norm[(2048, 112)](X, Y, 64, 262144, 128, num_warps=4, num_stages=1)
    return Y
