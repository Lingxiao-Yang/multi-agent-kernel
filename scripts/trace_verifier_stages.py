"""Run the generated verifier once and record which stage is active."""
import time
from contextlib import contextmanager
from pathlib import Path


def main():
    directory = Path(__file__).parent
    active = 'process'
    with (directory/'stages.tsv').open('w', buffering=1) as trace:
        @contextmanager
        def stage(name):
            nonlocal active
            previous, active = active, name
            trace.write(f'{time.monotonic()} {name}\n')
            try:
                yield
            finally:
                active = previous
                trace.write(f'{time.monotonic()} {previous}\n')

        with stage('imports'):
            path = directory/'original_test.py'
            body = {'__file__':str(path), '__name__':'original_test'}
            exec(compile(path.read_bytes(), str(path), 'exec'), body)
        tasks = body['kernelbench']
        synchronize = body['torch'].cuda.synchronize

        def measured(name, fn, *, gpu=False):
            def call(*args, **kwargs):
                if name == 'kernel':
                    synchronize()  # Finish the input snapshots before timing the kernel.
                with stage(name):
                    result = fn(*args, **kwargs)
                    if gpu:
                        synchronize()
                    return result
            return call

        tasks.upstream = measured('version', tasks.upstream)
        tasks.load_module = measured('imports', tasks.load_module)
        inputs = measured('inputs', tasks.task_inputs, gpu=True)

        def task_inputs(*args, **kwargs):
            model, arguments, buffers = inputs(*args, **kwargs)
            return measured('reference', model, gpu=True), arguments, buffers

        tasks.task_inputs = task_inputs
        body['kernel_function'] = measured('kernel', body['kernel_function'], gpu=True)
        body['check_forward'] = measured('comparison', body['check_forward'], gpu=True)
        body['test_kernel']()
        print('PASS')


if __name__ == '__main__':
    main()
