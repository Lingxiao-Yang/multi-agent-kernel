"""No automatic HTTP retries: every request consumes a visible model call."""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
import re
from dataclasses import dataclass
from http.client import IncompleteRead
from pathlib import Path

from .budget import BudgetExceeded, LedgerError, interruption
from .records import event


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    metadata: dict


class ChatProvider:
    name = 'llm'

    def __init__(self, model, endpoint, max_output_tokens=4096, temperature=0.7,
                 api_key_env='OPENAI_API_KEY', input_price=None, output_price=None, dollar_cap=None):
        if not model or not endpoint:
            raise ValueError('Explicit pinned model and endpoint are required')
        if not endpoint.startswith('https://') and not endpoint.startswith('http://localhost:'):
            raise ValueError('Use HTTPS or a localhost inference endpoint')
        self.model, self.endpoint = model, endpoint.rstrip('/')
        self.max_output_tokens, self.temperature = max_output_tokens, temperature
        self.key = os.environ.get(api_key_env, '')
        self.input_price, self.output_price, self.dollar_cap = input_price, output_price, dollar_cap
        self.spent = 0.
        self.completed_cost = 0.
        if dollar_cap is not None and (input_price is None or output_price is None):
            raise ValueError('A dollar cap requires explicit input/output token prices')

    def complete(self, system, context, stage, seed):
        messages = [{'role':'system', 'content':system},
                    {'role':'user', 'content':json.dumps({'stage':stage, **context}, sort_keys=True)}]
        return self.send_messages(messages,stage,seed,json_output=True)

    def send_messages(self, messages, stage, seed, json_output=False, max_tokens=None):
        output_cap = self.max_output_tokens if max_tokens is None else min(max_tokens,self.max_output_tokens)
        # UTF-8 byte count is a conservative input reservation, not reported usage.
        input_reservation = len(json.dumps(messages).encode('utf-8')) + 256
        reserved_cost = None
        if self.dollar_cap is not None:
            reserved_cost = (input_reservation*self.input_price + output_cap*self.output_price)/1e6
            if self.spent + reserved_cost > self.dollar_cap:
                raise RuntimeError('dollar_cap_reservation')
        payload = {'model':self.model, 'messages':messages, 'max_completion_tokens':output_cap}
        if json_output:
            payload['response_format'] = {'type':'json_object'}
        if self.temperature is not None:
            payload['temperature'] = self.temperature
        # Independent runs use API seed when the chosen endpoint supports it.
        payload['seed'] = seed
        request = urllib.request.Request(self.endpoint + '/chat/completions',
                  data=json.dumps(payload).encode(), headers={'Content-Type':'application/json',
                  'Authorization':'Bearer '+self.key})
        start = time.perf_counter()
        # Reserve on send; an ambiguous network failure may still incur charges.
        if reserved_cost is not None:
            self.spent += reserved_cost
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            body=exc.read().decode('utf-8',errors='replace')
            try:
                detail=json.loads(body).get('error',{})
            except json.JSONDecodeError:
                detail={}
            # Definite request rejection is not an ambiguous successful inference.
            if exc.code in [400,401,403,404,429] and reserved_cost is not None:
                self.spent -= reserved_cost
            message=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',str(detail.get('message','')))
            raise RuntimeError(f'HTTP {exc.code}: {detail.get("code", "unknown")} {message[:500]}') from None
        usage = data.get('usage')
        if not usage or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            raise RuntimeError('missing_usage: stop rather than invent token totals')
        actual_cost = None
        if self.input_price is not None and self.output_price is not None:
            actual_cost = (usage['prompt_tokens']*self.input_price + usage['completion_tokens']*self.output_price)/1e6
            self.completed_cost += actual_cost
            if reserved_cost is not None:
                self.spent += actual_cost - reserved_cost
            else:
                self.spent += actual_cost
        return Reply(data['choices'][0]['message']['content'], usage['prompt_tokens'], usage['completion_tokens'],
                     data.get('model', self.model), {'request_id':data.get('id'),
                     'system_fingerprint':data.get('system_fingerprint'), 'usage':usage,
                     'elapsed_s':time.perf_counter()-start, 'estimated_usd':actual_cost})


class MeteredChat(ChatProvider):
    def __init__(self, ledger, directory, **kwargs):
        super().__init__(**kwargs)
        self.ledger, self.directory = ledger, Path(directory)
        self.failure = None

    def send_messages(self, messages, stage, seed, json_output=False, max_tokens=None):
        if self.failure is not None:
            raise self.failure
        try:
            return self._send_messages(messages,stage,seed,json_output,max_tokens)
        except (LedgerError, OSError) as exc:
            self.failure = exc
            raise

    def _send_messages(self, messages, stage, seed, json_output, max_tokens):
        stopped = interruption(self.directory)
        if stopped:
            raise RuntimeError(stopped)
        reserved = len(json.dumps(messages).encode())+256
        output_cap = self.max_output_tokens if max_tokens is None else min(max_tokens,self.max_output_tokens)
        log = self.directory/'model_events.jsonl'
        try:
            number = self.ledger.reserve(stage,reserved,output_cap)
        except BudgetExceeded as exc:
            event(log,'budget_stop',stage=stage,reason=str(exc),budget=self.ledger.summary())
            raise
        event(log,'request',id=number,stage=stage,seed=seed,messages=messages,budget=self.ledger.summary())
        try:
            reply = super().send_messages(messages,stage,seed,json_output,output_cap)
        except Exception as exc:
            if str(exc).startswith(('HTTP 400:', 'HTTP 401:', 'HTTP 403:', 'HTTP 404:', 'HTTP 429:')):
                self.ledger.settle(number,0,0)
            reason = 'transport' if isinstance(exc, (OSError, IncompleteRead)) else 'provider_error'
            event(log,'request_error',id=number,reason=reason,error=str(exc))
            raise
        self.ledger.settle(number,reply.input_tokens,reply.output_tokens)
        event(log,'response',id=number,stage=stage,model=reply.model,content=reply.text,
              metadata=reply.metadata,budget=self.ledger.summary())
        if reply.model != self.model:
            event(log,'request_error',id=number,reason='model_changed',error='unexpected model snapshot')
            raise RuntimeError(f'resolved model {reply.model} differs from snapshot {self.model}')
        return reply
