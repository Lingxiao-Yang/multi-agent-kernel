"""Minimal profiler range excluding compilation and input generation."""
from .core import baseline_config, cases
from .bench import make_inputs, prepare


def main():
    import torch
    case=next(c for c in cases() if c['id']=='softmax_small')
    inputs=make_inputs(case,101)
    config=baseline_config(case['family']); config['fused']=True
    fn,_=prepare(case,config,inputs)
    fn(); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    fn(); torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


if __name__=='__main__':
    main()
