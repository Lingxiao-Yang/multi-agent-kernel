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


def cases():
    return read_json(ROOT / "configs/cases.json")


def baseline_config(family):
    return {"num_warps": 4, "num_stages": 1, "block": 256,
            "block_m": 16, "block_n": 16, "block_k": 32,
            "fused": False, "tensor_core": False}


def validate_config(config):
    if config.get('mode') == 'source-v1':
        from .source_space import validate_plan
        return validate_plan(config)
    expected = baseline_config("matmul")
    if set(config) != set(expected):
        raise ValueError("config must contain exactly " + str(sorted(expected)))
    domains = {"num_warps": [1, 2, 4, 8], "num_stages": [1, 2, 3, 4],
               "block": [64, 128, 256, 512, 1024],
               "block_m": [16, 32, 64], "block_n": [16, 32, 64],
               "block_k": [16, 32, 64]}
    for name, values in domains.items():
        if type(config[name]) is not int or config[name] not in values:
            raise ValueError(f"unsupported {name}: {config[name]}")
    for name in ("fused", "tensor_core"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    return config


def launch_contract(family):
    """Common API semantics, not optimization recommendations or knowledge cards."""
    common = ['num_warps', 'num_stages']
    if family == 'matmul':
        active = common + ['block_m', 'block_n', 'block_k', 'tensor_core']
        entry = {'tensor_core=false':'matmul_simt', 'tensor_core=true':'matmul_tensor'}
        semantics = 'shape is [M,N,K], inputs A[M,K], B[K,N], output Y[M,N]. BM/BN/BK are tile parameters; BK is used only by matmul_tensor.'
    elif family in ['softmax','layernorm']:
        active = common + ['block','fused']
        entry = {'fused=false':[family+'_stats',family+'_apply'], 'fused=true':family+'_fused'}
        semantics = 'shape is [rows,columns]. Row reductions use next_power_of_2(columns). block controls the apply launch only when fused=false.'
    else:
        active = common + ['block','fused']
        entry = {'fused=false':['bias_add','relu'], 'fused=true':'bias_relu'}
        semantics = 'shape is [rows,columns]; bias has [columns]. block is the number of flat output elements per program.'
    return {'active_config_fields':active,'ignored_config_fields':sorted(set(baseline_config(family))-set(active)),
            'entry_points':entry,'semantics':semantics,
            'note':'Ignore unrelated family functions in the shared device module. Changing an ignored field does not change this operation.'}


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
