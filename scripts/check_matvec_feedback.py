"""Check the real verifier deadline and hardware feedback on archived matvec inputs."""
import argparse
from pathlib import Path
import time

from kernel_study.judge.evaluate import ncu_evidence
from kernel_study.records import ROOT, digest, write_json
from kernel_study.tasks import kernelbench, starts


def verify(worker, source, case):
    text = source.read_text()
    worker._write_files(text, [kernelbench.test_source(case)])
    begin = time.monotonic()
    ok, stdout, stderr = worker._run_test()
    return dict(passed=ok, seconds=time.monotonic()-begin, timeout_s=worker.test_timeout_s,
                source_hash=digest(text), stdout=stdout, stderr=stderr)


def main():
    from triton_kernel_agent.worker import VerificationWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = kernelbench.experiment()['cases']['4']
    baseline = starts.start_path(case)
    result = {}
    for name, path in [('baseline',baseline),('proposal_005',ROOT/'tests/judge/fixtures/matvec_dtype_constants.py')]:
        work = output/name
        work.mkdir(exist_ok=True)
        result[name] = verify(VerificationWorker(0,work,work), path, case)
        write_json(output/'verification.json', result)
    feedback = ncu_evidence(baseline, kernelbench.task_path(case), output/'ncu')
    write_json(output/'hardware.json', feedback)
    print('NCU metrics:', sum(len(metrics) for metrics in feedback['metrics'].values()), flush=True)


if __name__ == '__main__':
    main()
