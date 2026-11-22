"""Profile one public GELU launch after warming up the kernel."""
from pathlib import Path
from tempfile import TemporaryDirectory

from .public_cases import experiment, local_forward, start, task_inputs


def main():
    import torch
    case = experiment()['cases']['26']
    _, _, inputs = task_inputs(case, 101)
    source, config = start(case)
    with TemporaryDirectory() as directory:
        path = Path(directory)/'device.py'
        path.write_text(source)
        forward = local_forward(case, path, config, inputs)
        forward()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        forward()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__=='__main__':
    main()
