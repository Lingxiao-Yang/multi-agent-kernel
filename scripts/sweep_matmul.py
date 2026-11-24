"""Measure launch parameters of the frozen IEEE FP32 matrix multiplication."""
import argparse
from itertools import product
from pathlib import Path
import random

from kernel_study.campaign import launch_sweep
from kernel_study.campaign.jobs import register
from kernel_study.campaign.preflight import check_device
from kernel_study.records import digest, write_json
from kernel_study.tasks import kernelbench, starts


BASELINE = (32,32,32,4,1)


def configurations():
    grid = list(product((32,64,128), (32,64,128), (16,32), (4,8), (1,3)))
    grid.remove(BASELINE)
    random.Random(771).shuffle(grid)
    return [BASELINE, *grid]


def label(config):
    return 'm{}-n{}-k{}-w{}-s{}'.format(*config)


def launch(config):
    bm,bn,bk,warps,stages = config
    return (f'    matmul_tensor[({2048//bm}, {4096//bn})](A, B, Y, 2048, 4096, 8192, '
            f'{bm}, {bn}, {bk}, num_warps={warps}, num_stages={stages})\n')


def variant(source, config):
    original = launch(BASELINE)
    if source.count(original)!=1:
        raise ValueError('matmul launch differs from the frozen source')
    return source.replace(original,launch(config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    plan = kernelbench.experiment()
    case = plan['cases']['2']
    grid = configurations()
    protocol = launch_sweep.protocol(plan)
    register(output,'manifest',dict(device=check_device(plan),case=case,protocol=protocol,grid=grid,
             frozen_source_hash=digest(starts.start(case)),parameters=['BM','BN','BK','num_warps','num_stages'],
             finalist_count=8,finalist_repeats=3,compile_mode=plan['compile_mode']))
    result = launch_sweep.run_grid(output,case,protocol,grid,BASELINE,label,variant)
    write_json(output/'summary.json',result)


if __name__=='__main__':
    main()
