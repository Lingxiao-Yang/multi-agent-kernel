"""One candidate per process, keeping references/timers outside model code."""
import argparse
import time
import traceback
from .core import read_json, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("request")
    p.add_argument("output")
    args = p.parse_args()
    start = time.perf_counter()
    try:
        from .bench import evaluate
        result = evaluate(read_json(args.request))
    except Exception as exc:
        result = {"status": "incorrect" if isinstance(exc, AssertionError) else "error",
                  "error_type": type(exc).__name__, "error": str(exc),
                  "traceback": traceback.format_exc(), "worker_seconds": time.perf_counter() - start}
    write_json(args.output, result)


if __name__ == "__main__":
    main()
