"""No automatic HTTP retries: every request consumes a visible model call."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


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
        if dollar_cap is not None and (input_price is None or output_price is None):
            raise ValueError('A dollar cap requires explicit input/output token prices')

    def complete(self, system, context, stage, seed):
        messages = [{'role':'system', 'content':system},
                    {'role':'user', 'content':json.dumps({'stage':stage, **context}, sort_keys=True)}]
        # UTF-8 byte count is a conservative input reservation, not reported usage.
        input_reservation = len(json.dumps(messages).encode('utf-8')) + 256
        reserved_cost = None
        if self.dollar_cap is not None:
            reserved_cost = (input_reservation*self.input_price + self.max_output_tokens*self.output_price)/1e6
            if self.spent + reserved_cost > self.dollar_cap:
                raise RuntimeError('dollar_cap_reservation')
        payload = {'model':self.model, 'messages':messages, 'max_completion_tokens':self.max_output_tokens,
                   'response_format':{'type':'json_object'}}
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
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.load(response)
        usage = data.get('usage')
        if not usage or 'prompt_tokens' not in usage or 'completion_tokens' not in usage:
            raise RuntimeError('missing_usage: stop rather than invent token totals')
        actual_cost = None
        if self.input_price is not None and self.output_price is not None:
            actual_cost = (usage['prompt_tokens']*self.input_price + usage['completion_tokens']*self.output_price)/1e6
            if reserved_cost is not None:
                self.spent += actual_cost - reserved_cost
            else:
                self.spent += actual_cost
        return Reply(data['choices'][0]['message']['content'], usage['prompt_tokens'], usage['completion_tokens'],
                     data.get('model', self.model), {'request_id':data.get('id'),
                     'system_fingerprint':data.get('system_fingerprint'), 'usage':usage,
                     'elapsed_s':time.perf_counter()-start, 'estimated_usd':actual_cost})


class ScriptedProvider:
    """Infrastructure control. It is never evidence about LLM specialization."""
    name = 'scripted-control'
    model = 'none'

    def complete(self, system, context, stage, seed):
        if stage == 'implement':
            config = dict(context['incumbent']['config'])
            attempt = context['attempt']
            family = context['case']['family']
            if family == 'matmul':
                config['tensor_core'] = True
                config['block_m'] = [16,32,64,32][(attempt-1)%4]
                config['block_n'] = [16,32,64,64][(attempt-1)%4]
                config['num_stages'] = [1,2,3,2][(attempt-1)%4]
            else:
                config['fused'] = True
                config['num_warps'] = [4, 8, 2, 4][(attempt-1)%4]
                config['block'] = [256,512,128,1024][(attempt-1)%4]
            text = json.dumps({'config':config,'source':None,'rationale':'Predefined engineering control; no LLM inference.'})
        else:
            text = json.dumps({'note':'Scripted plumbing check; no scientific role/knowledge comparison.'})
        return Reply(text, 0, 0, 'none', {'synthetic':True, 'estimated_usd':0})
