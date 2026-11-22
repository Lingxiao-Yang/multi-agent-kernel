"""Build the shared device program for each public task."""
import ast
import math
from pathlib import Path

from .kernelbench import layout, load_module, matvec_path

MODE = 'source-v1'


def load_native(path):
    return load_module(path).kernel_function


def device_function(path,name):
    text = Path(path).read_text()
    fn = next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    return '@triton.jit\n' + ast.get_source_segment(text,fn)


def start(case):
    family = case['family']
    shape = case['output_shape']
    _, sizes, _ = layout(case)
    m,n,total = (sizes[k] for k in ('M','N','TOTAL'))
    public = Path(__file__).with_name('kernels.py')
    kernels, launches = [], []
    def add(path, name, grid, args, stages=1):
        source = device_function(path,name)
        kernels.append(source)
        launches.append(dict(kernel=name,grid=grid,args=args,num_warps=4,num_stages=stages))
    if family == 'matvec':
        add(matvec_path(),'_matvec_kernel',[(m+127)//128],
            ['A','B','Y','M','K','K',1,1,1,1,1,128,256],2)
    elif family == 'matmul':
        add(public,'matmul_tensor',[(m+31)//32,(n+31)//32],['A','B','Y','M','N','K',32,32,32])
    elif family in ('softmax','layernorm'):
        groups = (n+1023)//1024
        flag = int(family=='softmax')
        if flag:
            add(public,'row_partials',[groups,m],['X','S','ROWS','N',groups,1024,flag])
        else:
            add(public,'layernorm_partials',[groups,m],['X','S','ROWS','N',groups,1024])
        add(public,'row_combine',[m],['S','ROWS','N',groups,1024,1 << (groups-1).bit_length(),flag])
        operands = ['X'] if flag else ['X','W','B']
        add(public,family+'_apply',[(total+2047)//2048],operands+['S','Y','TOTAL','N',2048])
    elif family == 'gelu':
        add(public,'gelu',[(total+2047)//2048],['X','Y','TOTAL',2048])
    elif family == 'rmsnorm':
        spatial = math.prod(shape[2:])
        add(public,'rms_norm',[(spatial+127)//128,m],['X','Y',shape[1],spatial,128])
    else:
        raise ValueError(family)
    return 'import triton\nimport triton.language as tl\n\n' + '\n\n'.join(kernels)+'\n', {
        'mode':MODE,'launches':launches}
