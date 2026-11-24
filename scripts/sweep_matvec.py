"""Measure launch configurations of the frozen matvec implementation on one A100."""
import argparse
from itertools import product
from pathlib import Path
import random

from kernel_study.campaign import launch_sweep
from kernel_study.campaign.jobs import register
from kernel_study.campaign.launch_sweep import matvec_label as label
from kernel_study.campaign.preflight import check_device
from kernel_study.records import digest, write_json
from kernel_study.tasks import kernelbench, starts


BASELINE = (128, 256, 4, 2)


def configurations():
    grid = list(product((1, 2, 4, 8, 16, 32, 64, 128), (64, 128, 256, 512),
                        (1, 2, 4, 8), (1, 2, 3, 4)))
    grid.remove(BASELINE)
    random.Random(771).shuffle(grid)
    return [BASELINE, *grid]


def variant(source, config):
    for fmt, old, new in zip(('    BLOCK_M = {}\n', '    BLOCK_K = {}\n',
                               '        num_warps={},\n', '        num_stages={},\n'), BASELINE, config):
        old_line = fmt.format(old)
        if source.count(old_line) != 1:
            raise ValueError(f'launch parameter differs from frozen source: {old_line.strip()}')
        source = source.replace(old_line, fmt.format(new))
    return source


def run(output, case, protocol, grid):
    summary = launch_sweep.run_grid(output,case,protocol,grid,BASELINE,label,variant)
    write_json(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = kernelbench.experiment()
    case = plan['cases']['4']
    grid = configurations()
    protocol = launch_sweep.protocol(plan)
    register(output, 'manifest', dict(device=check_device(plan), case=case, protocol=protocol,
             frozen_source_hash=digest(starts.start(case)), grid=grid, parameters=['BLOCK_M','BLOCK_K','num_warps','num_stages'],
             finalist_count=8, finalist_repeats=3, compile_mode=plan['compile_mode']))
    run(output, case, protocol, grid)


if __name__ == '__main__':
    main()
