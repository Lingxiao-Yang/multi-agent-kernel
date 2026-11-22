from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("single", "single_knowledge", "multi", "multi_knowledge")


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


@dataclass
class Budget:
    candidate_cap: int = 20
    call_cap: int = 100
    token_safety_cap: int = 500000
    attempts: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self):
        return self.input_tokens + self.output_tokens

    def stop_reason(self):
        if self.tokens >= self.token_safety_cap:
            return "token_safety_cap"
        if self.attempts >= self.candidate_cap:
            return "candidate_cap"
        if self.calls >= self.call_cap:
            return "call_cap"
        return None

    def reserve_attempt(self):
        if self.stop_reason():
            raise RuntimeError(self.stop_reason())
        self.attempts += 1

    def reserve_call(self):
        # Candidate count is already reserved before its model calls.
        if self.calls >= self.call_cap or self.tokens >= self.token_safety_cap:
            raise RuntimeError("model_budget_exhausted")
        self.calls += 1

    def charge_tokens(self, input_tokens, output_tokens):
        if min(input_tokens, output_tokens) < 0:
            raise ValueError("negative token count")
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def snapshot(self):
        return {**asdict(self), "tokens": self.tokens}


def environment():
    import sys
    result = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch
        import triton
        result.update(torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda)
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            result.update(gpu=p.name, memory_bytes=p.total_memory,
                          capability=list(torch.cuda.get_device_capability(0)), sm_count=p.multi_processor_count)
    except ImportError:
        pass
    return result
