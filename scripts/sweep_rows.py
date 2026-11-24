"""Measure the launch parameters of the frozen softmax and LayerNorm kernels."""
import argparse
from functools import partial
from itertools import product
from math import comb, prod
from pathlib import Path
import random
from statistics import median

from kernel_study.campaign import launch_sweep
from kernel_study.campaign.jobs import register
from kernel_study.campaign.preflight import check_device
from kernel_study.records import digest, read_json, write_json
from kernel_study.tasks import kernelbench, starts


BASELINE = (1024,2048,4,4,1)
BLOCKS = (512,1024,2048,4096)
WARPS = (1,2,4,8)
STAGES = (1,2,3,4)
PILOT = (BASELINE[:-1], (512,1024,2,2), (4096,4096,8,8))


def configurations(stages):
    grid = list(product(BLOCKS,BLOCKS,WARPS,WARPS,stages))
    grid.remove(BASELINE)
    random.Random(771).shuffle(grid)
    return [BASELINE,*grid]


def label(config):
    return 'r{}-a{}-rw{}-aw{}-s{}'.format(*config)


def launches(case, config):
    reduce_block,apply_block,reduce_warps,apply_warps,stages = config
    rows,*dimensions = case['output_shape']
    width = prod(dimensions)
    total = rows*width
    groups = (width+reduce_block-1)//reduce_block
    padded = 1 << (groups-1).bit_length()
    apply_grid = (total+apply_block-1)//apply_block
    softmax = case['family']=='softmax'
    partial_kernel = 'row_partials' if softmax else 'layernorm_partials'
    partial_flag = ', 1' if softmax else ''
    apply_args = 'X, S, Y' if softmax else 'X, weight, bias, S, Y'
    reduce_params = f'num_warps={reduce_warps}, num_stages={stages}'
    apply_params = f'num_warps={apply_warps}, num_stages={stages}'
    return (
        f'    S = Y.new_empty(({2*rows*(groups+1)},), dtype=torch.float32)\n'
        f'    {partial_kernel}[({groups}, {rows})](X, S, {rows}, {width}, {groups}, {reduce_block}{partial_flag}, {reduce_params})\n'
        f'    row_combine[({rows},)](S, {rows}, {width}, {groups}, {reduce_block}, {padded}, {int(softmax)}, {reduce_params})\n'
        f'    {case["family"]}_apply[({apply_grid},)]({apply_args}, {total}, {width}, {apply_block}, {apply_params})\n'
    )


def variant(case, source, config):
    frozen = launches(case,BASELINE)
    if source.count(frozen)!=1:
        raise ValueError('row launch configuration differs from the frozen source')
    return source.replace(frozen,launches(case,config))


def stage_pilot(output, case, settings):
    sources = output/'sources'
    sources.mkdir(parents=True,exist_ok=True)
    original = starts.start(case)
    checks = []
    threshold = settings['improvement_threshold']
    for layout in PILOT:
        paths = {}
        for stage in STAGES:
            config = (*layout,stage)
            paths[stage] = sources/(label(config)+'.py')
            paths[stage].write_text(variant(case,original,config))
        for stage in STAGES[1:]:
            name = 'pilot-'+label((*layout,stage))
            result = launch_sweep.measure(output,name,case,settings,paths[stage],paths[1])
            row = dict(layout=list(layout),stage=stage,status=result['status'],speedup=result.get('paired_speedup'))
            if result['status']=='ok' and result['paired_speedup'] < 1/threshold:
                reverse = launch_sweep.measure(output,name+'-reverse',case,settings,paths[1],paths[stage])
                row.update(reverse_status=reverse['status'],reverse_speedup=reverse.get('paired_speedup'))
                row['keep_stages'] = reverse['status']!='ok' or reverse.get('confirmed',False)
            else:
                row['keep_stages'] = result['status']!='ok' or result.get('confirmed',False)
            checks.append(row)
    levels = list(STAGES) if any(r['keep_stages'] for r in checks) else [1]
    return dict(stages=levels,checks=checks,threshold=threshold)


def random_four(latencies, population, confirmed):
    """Exact order statistics for four distinct uniformly sampled configurations."""
    ordered = sorted(latencies)
    draws = comb(population,4)
    def quantile(q):
        return next((ms for i,ms in enumerate(ordered,1) if comb(population-i,4) <= (1-q)*draws),None)
    return dict(median_ms=quantile(.5),p25_ms=quantile(.25),p75_ms=quantile(.75),
                valid=len(ordered),population=population,draw_size=4,
                probability_no_valid=comb(population-len(ordered),4)/draws,
                probability_confirmed_gain=1-comb(population-confirmed,4)/draws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['23','40'],required=True)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    plan = kernelbench.experiment()
    case = plan['cases'][args.case]
    settings = launch_sweep.protocol(plan)
    register(output,'setup',dict(case=case,device=check_device(plan),protocol=settings,frozen_source_hash=digest(starts.start(case)),
             parameters=['reduce_block','apply_block','reduce_warps','apply_warps','stages'],blocks=BLOCKS,warps=WARPS,
             pilot_layouts=PILOT,pilot_stages=STAGES,finalist_count=8,finalist_repeats=3))
    pilot = stage_pilot(output/'pilot',case,settings)
    write_json(output/'stages.json',pilot)
    grid = configurations(pilot['stages'])
    register(output,'manifest',dict(grid=grid,stages=pilot['stages']))
    result = launch_sweep.run_grid(output,case,settings,grid,BASELINE,label,partial(variant,case))
    records = [read_json(output/('screen-'+label(c)+'.result.json')) for c in grid]
    result['random_best_of_four'] = random_four([r['candidate']['median_ms'] for r in records if r['status']=='ok'],
                                               len(grid),result['confirmed_in_screen'])
    checks = result['best'].get('checks',[])
    best_times = [read_json(output/r['result_file'])['candidate']['median_ms'] for r in checks]
    result['best_parameter_ms'] = median(best_times) if best_times else records[0]['candidate']['median_ms']
    write_json(output/'summary.json',result)


if __name__=='__main__':
    main()
