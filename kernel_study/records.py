"""Experiment records, environment and registered code hashes."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def digest(value):
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def event(path, kind, **data):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_utc": utcnow(), "event": kind, **data}, allow_nan=False) + "\n")
        handle.flush()


def environment():
    import sys
    result = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        import triton
        result.update(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            result.update(gpu=p.name, uuid=str(p.uuid), node=platform.node(), memory_bytes=p.total_memory,
                          capability=list(torch.cuda.get_device_capability(0)), sm_count=p.multi_processor_count)
    except ImportError:
        pass
    return result


def code_hashes():
    package = ROOT/'kernel_study'
    files = [path for path in package.glob('*.py') if path.name != 'cli.py']
    files.extend(package/'campaign'/name for name in ('__init__.py', 'jobs.py', 'preflight.py', 'roles.py'))
    for directory in ('tasks', 'judge', 'agents'):
        files.extend((package/directory).rglob('*.py'))
    files.extend(ROOT/p for p in ('configs/agent_baseline.json', 'configs/protocol.json', 'configs/greedy.json', 'knowledge/cards.json'))
    return {path.relative_to(ROOT).as_posix():digest(path.read_text(encoding='utf-8')) for path in sorted(files)}
