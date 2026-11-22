"""Profile one public GELU launch after warming up the kernel."""
from pathlib import Path
from tempfile import TemporaryDirectory

from kernel_study.judge.interface import local_forward
from kernel_study.tasks.kernelbench import experiment
from kernel_study.tasks.starts import start


def main():
    import torch
    case = dict(experiment()['cases']['26'], output_shape=[1, 2048])
    x = torch.zeros(case['output_shape'], device='cuda', dtype=torch.float32)
    source, config = start(case)
    with TemporaryDirectory() as directory:
        path = Path(directory)/'device.py'
        path.write_text(source)
        forward = local_forward(case, path, config, [x])
        forward()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        forward()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__=='__main__':
    main()
