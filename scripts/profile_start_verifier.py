"""Time three fresh upstream verification processes on the same matvec."""
import argparse
from pathlib import Path
import socket
import sys
import time

from kernel_study.records import digest, write_json
from kernel_study.tasks import kernelbench, starts


def stage_times(trace, start, end):
    times = {}
    active = 'process'
    if trace.exists():
        for line in trace.read_text().splitlines():
            timestamp, name = line.split()
            now = float(timestamp)
            times[active] = times.get(active, 0) + now - start
            active, start = name, now
    times[active] = times.get(active, 0) + end - start
    return times, active


def main():
    import torch
    import triton
    from triton_kernel_agent.worker import VerificationWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    case = kernelbench.experiment()['cases']['4']
    kernelbench.load_module(kernelbench.task_path(case))
    source = starts.start(case)
    test = kernelbench.test_source(case)
    probe = Path(__file__).with_name('trace_verifier_stages.py').read_text()
    write_json(out/'environment.json', dict(node=socket.gethostname(), python=sys.executable,
               prefix=sys.prefix, torch=torch.__file__, triton=triton.__file__, project=kernelbench.__file__,
               device=str(torch.cuda.get_device_properties(0).uuid), source_hash=digest(source),
               test_hash=digest(test), probe_hash=digest(probe)))
    for attempt in range(1, 4):
        work = out/str(attempt)
        work.mkdir()
        worker = VerificationWorker(0, work, work, test_timeout_s=30)
        worker._write_files(source, [probe])
        (work/'original_test.py').write_text(test)
        start = time.monotonic()
        ok, stdout, stderr = worker._run_test()
        end = time.monotonic()
        stages, last = stage_times(work/'stages.tsv', start, end)
        write_json(work/'verification.json', dict(passed=ok, seconds=end-start, timeout_s=worker.test_timeout_s,
                   stages=stages, last_stage=last, stdout=stdout, stderr=stderr))
        print(f'{out.name} {attempt}: {end-start:.3f}s, passed={ok}, last={last}', flush=True)


if __name__ == '__main__':
    main()
