"""Remeasure the two archived matvec finalists with the registered public timing protocol."""
import argparse
from pathlib import Path
import random

from kernel_study.campaign.preflight import check_device
from kernel_study.judge.correctness import check_forward
from kernel_study.judge.timing import first_call, time_pair
from kernel_study.records import digest, write_json
from kernel_study.tasks import kernelbench
from scripts.measure_matvec_idle import archived_sources


def measure(programs, libraries, protocol):
    rng = random.Random(771)
    cases = [(name,ref) for name in programs for ref in ('self','eager','compile')]
    cases.append(('greedy','start'))
    rows = []
    for repeat in range(3):
        order = cases.copy()
        rng.shuffle(order)
        for name,ref in order:
            left = programs[name]
            right = left if ref=='self' else {**programs,**libraries}[ref]
            a,b = time_pair(left,right,protocol)
            rows.append(dict(repeat=repeat,program=name,reference=ref,measured=a,comparator=b,
                             speedup=b['median_ms']/a['median_ms']))
    return rows


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    plan = kernelbench.experiment()
    device = check_device(plan)
    case = plan['cases']['4']
    seed = plan['final_validation_seeds'][0]
    paths = archived_sources(args.archive)
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.no_grad():
        model, inputs, buffers = kernelbench.task_inputs(case,seed)
        modules = {k:kernelbench.load_module(p).kernel_function for k,p in paths.items()}
        programs = {k:(lambda fn=fn: fn(*buffers)) for k,fn in modules.items()}
        expected = model(*inputs)
        checks = {k:check_forward(fn,buffers,expected,case) for k,fn in modules.items()}
        del expected
        compiled = torch.compile(model,mode=plan['compile_mode'],fullgraph=True)
        libraries = dict(eager=lambda: model(*inputs),compile=lambda: compiled(*inputs))
        value, compile_first_seconds = first_call(libraries['compile'])
        del value
        rows = measure(programs,libraries,plan['timing'])
    write_json(args.output,dict(device=device,seed=seed,protocol=plan['timing'],
               source_hashes={k:digest(p.read_text()) for k,p in paths.items()},
               checks=checks,compile_first_seconds=compile_first_seconds,pairs=rows))


if __name__=='__main__':
    main()
