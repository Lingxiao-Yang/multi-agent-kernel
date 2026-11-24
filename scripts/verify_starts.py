"""Check one or all frozen starts with the upstream verifier."""
import argparse
from pathlib import Path
import time

from kernel_study.records import digest, write_json
from kernel_study.tasks import kernelbench, starts


def verify(worker, source, case):
    text = source.read_text()
    worker._write_files(text, [kernelbench.test_source(case)])
    begin = time.monotonic()
    ok, stdout, stderr = worker._run_test()
    return dict(passed=ok, seconds=time.monotonic()-begin, timeout_s=worker.test_timeout_s,
                source_hash=digest(text), stdout=stdout, stderr=stderr)


def deadline_status(row):
    if row['passed']:
        return 0 if row['seconds'] <= 25 else 75
    return 75 if row['stderr'].strip() == 'Test execution timed out after 30 seconds' else 1


def check_start(worker, source, case, versions):
    attempts = []
    for i in range(1, 4):
        row = verify(worker, source, case)
        attempts.append(row)
        write_json(worker.workdir/'verification.json', dict(row, attempts=attempts, **versions))
        print(f"Startup verification {i}: {row['seconds']:.3f} seconds, passed={row['passed']}", flush=True)
        status = deadline_status(row)
        if status != 75:
            return status
    return 75


def main():
    from triton_kernel_agent.worker import VerificationWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--start-deadline', action='store_true')
    mode.add_argument('--all-starts', action='store_true')
    cases = kernelbench.experiment()['cases']
    parser.add_argument('--case', choices=cases, default='4')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.all_starts:
        passed = []
        for case_id, case in cases.items():
            work = output/case_id
            work.mkdir()
            worker = VerificationWorker(0, work, work, test_timeout_s=30)
            row = verify(worker, starts.start_path(case), case)
            write_json(work/'verification.json', row)
            print(f"{case['family']}: {row['seconds']:.3f} seconds, passed={row['passed']}", flush=True)
            passed.append(row['passed'])
        raise SystemExit(0 if all(passed) else 1)
    import torch
    import triton
    case = cases[args.case]
    kernelbench.load_module(kernelbench.task_path(case))
    worker = VerificationWorker(0, output, output, test_timeout_s=30)
    versions = dict(torch=torch.__version__, triton=triton.__version__)
    raise SystemExit(check_start(worker, starts.start_path(case), case, versions))


if __name__ == '__main__':
    main()
