"""Measure launch configurations of the frozen matvec implementation on one A100."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
import random
from statistics import median

from kernel_study.campaign.jobs import register
from kernel_study.campaign.preflight import check_device
from kernel_study.judge.evaluate import run_worker
from kernel_study.records import ROOT, digest, read_json, write_json
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


def label(config):
    return 'm{}-k{}-w{}-s{}'.format(*config)


def run(output, case, protocol, grid):
    output.mkdir(parents=True, exist_ok=True)
    sources = output/'sources'
    sources.mkdir(exist_ok=True)
    original = starts.start(case)
    baseline = sources/(label(BASELINE)+'.py')
    counts = Counter()
    screening = []
    for config in grid:
        (sources/(label(config)+'.py')).write_text(variant(original, config))

    def measure(config, name, settings, final=False):
        path = output/(name+'.result.json')
        if path.is_file():
            return read_json(path)
        request = dict(case=case, protocol=settings, source=str(sources/(label(config)+'.py')),
                       incumbent_source=str(baseline), interface='native', final=final, library=final)
        return run_worker(request, output, name)

    for index, config in enumerate(grid, 1):
        name = 'screen-'+label(config)
        result = measure(config, name, protocol)
        counts[result['status']] += 1
        if config == BASELINE and result['status'] != 'ok':
            raise RuntimeError(f'baseline failed: see {name}.result.json')
        if config != BASELINE and result.get('confirmed'):
            screening.append((result['paired_speedup'], config))
        print(f'{index}/{len(grid)} {name} {result["status"]}', flush=True)

    ranked = sorted(screening, reverse=True)[:8]
    seeds = protocol['final_validation_seeds']
    finalists = []
    for _, config in ranked:
        checks = []
        for repeat in range(3):
            settings = dict(protocol, final_validation_seeds=seeds[repeat:]+seeds[:repeat])
            name = f'final-{repeat+1}-'+label(config)
            result = measure(config, name, settings, final=True)
            checks.append(dict(status=result['status'], confirmed=result.get('confirmed', False),
                               speedup=result.get('paired_speedup'), result_file=name+'.result.json'))
        if all(row['confirmed'] for row in checks):
            speedup = median(row['speedup'] for row in checks)
        else:
            speedup = None
        finalists.append(dict(config=list(config), checks=checks, speedup=speedup))
    eligible = [row for row in finalists if row['speedup'] is not None]
    best = max(eligible, key=lambda row: row['speedup']) if eligible else dict(config=list(BASELINE), speedup=1.)
    summary = dict(counts=dict(counts), confirmed_in_screen=len(screening), finalists=finalists, best=best)
    write_json(output/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = kernelbench.experiment()
    case = plan['cases']['4']
    grid = configurations()
    protocol = dict(read_json(ROOT/'configs/protocol.json'), **plan['timing'])
    protocol.update(validation_seeds=[101], final_validation_seeds=plan['final_validation_seeds'],
                    worker_timeout_s=600)
    register(output, 'manifest', dict(device=check_device(plan), case=case, protocol=protocol,
             frozen_source_hash=digest(starts.start(case)), grid=grid, parameters=['BLOCK_M','BLOCK_K','num_warps','num_stages'],
             finalist_count=8, finalist_repeats=3, compile_mode=plan['compile_mode']))
    run(output, case, protocol, grid)


if __name__ == '__main__':
    main()
