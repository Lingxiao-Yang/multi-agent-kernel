"""Screen launch configurations against a frozen source and recheck the fastest ones."""
from collections import Counter
from statistics import median

from ..judge.evaluate import run_worker
from ..records import ROOT, read_json
from ..tasks import starts


def matvec_label(config):
    return 'm{}-k{}-w{}-s{}'.format(*config)


def protocol(plan):
    settings = dict(read_json(ROOT/'configs/protocol.json'), **plan['timing'])
    return dict(settings, validation_seeds=plan['validation_seeds'], final_validation_seeds=plan['final_validation_seeds'],
                worker_timeout_s=600)


def measure(output, name, case, settings, source, incumbent, final=False):
    path = output/(name+'.result.json')
    if path.is_file():
        return read_json(path)
    request = dict(case=case, protocol=settings, source=str(source), incumbent_source=str(incumbent),
                   interface='native', final=final, library=final)
    return run_worker(request, output, name)


def run_grid(output, case, settings, grid, baseline, label, variant):
    sources = output/'sources'
    sources.mkdir(parents=True,exist_ok=True)
    original = starts.start(case)
    paths = {}
    for config in grid:
        paths[config] = sources/(label(config)+'.py')
        paths[config].write_text(variant(original,config))
    counts = Counter()
    screening = []
    for index, config in enumerate(grid,1):
        name = 'screen-'+label(config)
        result = measure(output,name,case,settings,paths[config],paths[baseline])
        counts[result['status']] += 1
        if config == baseline and result['status'] != 'ok':
            raise RuntimeError(f'baseline failed: see {name}.result.json')
        if config != baseline and result.get('confirmed'):
            screening.append((result['paired_speedup'],config))
        print(f'{index}/{len(grid)} {name} {result["status"]}',flush=True)

    seeds = settings['final_validation_seeds']
    finalists = []
    for _, config in sorted(screening,reverse=True)[:8]:
        checks = []
        for repeat in range(3):
            final_settings = dict(settings, final_validation_seeds=seeds[repeat:]+seeds[:repeat])
            name = f'final-{repeat+1}-'+label(config)
            result = measure(output,name,case,final_settings,paths[config],paths[baseline],final=True)
            checks.append(dict(status=result['status'], confirmed=result.get('confirmed',False),
                               speedup=result.get('paired_speedup'), result_file=name+'.result.json'))
        speedup = median(r['speedup'] for r in checks) if all(r['confirmed'] for r in checks) else None
        finalists.append(dict(config=list(config),checks=checks,speedup=speedup))
    eligible = [r for r in finalists if r['speedup'] is not None]
    best = max(eligible,key=lambda r:r['speedup']) if eligible else dict(config=list(baseline),speedup=1.)
    return dict(counts=dict(counts),confirmed_in_screen=len(screening),finalists=finalists,best=best)
