"""Profile one small launch of the registered GELU device kernel."""
def main():
    import torch
    from kernel_study.tasks.baselines.gelu import gelu

    x = torch.zeros(2048, device='cuda', dtype=torch.float32)
    y = torch.empty_like(x)
    def forward():
        gelu[(1,)](x, y, 2048, 2048, num_warps=4, num_stages=1)
    forward()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    try:
        forward()
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__=='__main__':
    main()
