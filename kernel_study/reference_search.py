"""Per-request accounting shared by the public single and greedy campaigns."""
import json
import sqlite3
from http.client import IncompleteRead
from contextlib import closing, contextmanager
from pathlib import Path

from .core import Budget, event
from .providers import ChatProvider


class Ledger:
    def __init__(self, directory, job, candidate_cap, call_cap, input_price, output_price, allowance):
        self.path = Path(directory)/'usage.sqlite3'
        self.job = job
        self.candidate_cap, self.call_cap = candidate_cap, call_cap
        self.input_price, self.output_price, self.allowance = input_price, output_price, allowance
        with self.transaction() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, job TEXT, stage TEXT, attempt INTEGER,
                committed REAL, completed REAL, input_tokens INTEGER, output_tokens INTEGER)''')

    @contextmanager
    def transaction(self):
        with closing(sqlite3.connect(self.path,timeout=30)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            yield db

    def cost(self, inputs, outputs):
        return (inputs*self.input_price + outputs*self.output_price)/1e6

    def reserve(self, stage, inputs, outputs):
        with self.transaction() as db:
            calls,attempts = db.execute('SELECT COUNT(*),COALESCE(SUM(attempt),0) FROM calls WHERE job=?',
                                        (self.job,)).fetchone()
            budget = Budget(candidate_cap=self.candidate_cap,call_cap=self.call_cap,attempts=attempts,calls=calls)
            proposal = stage in ('proposal','repair','implement')
            if proposal: budget.reserve_attempt()
            budget.reserve_call()
            committed, = db.execute('SELECT COALESCE(SUM(committed),0) FROM calls').fetchone()
            reserve = self.cost(inputs,outputs)
            if committed+reserve > self.allowance:
                raise RuntimeError('dollar_cap_reservation')
            return db.execute('INSERT INTO calls(job,stage,attempt,committed) VALUES(?,?,?,?)',
                              (self.job,stage,int(proposal),reserve)).lastrowid

    def settle(self, number, inputs, outputs):
        cost = self.cost(inputs,outputs)
        with self.transaction() as db:
            changed = db.execute('''UPDATE calls SET input_tokens=?,output_tokens=?,committed=?,completed=?
                                    WHERE id=? AND job=? AND completed IS NULL''',
                                 (inputs,outputs,cost,cost,number,self.job))
            if changed.rowcount != 1:
                raise ValueError('unreserved or already settled request')

    def summary(self):
        with self.transaction() as db:
            row = db.execute('''SELECT COUNT(*),SUM(attempt),SUM(input_tokens),SUM(output_tokens),
                                SUM(committed),SUM(completed) FROM calls WHERE job=?''',(self.job,)).fetchone()
        fields = ['calls','attempts','input_tokens','output_tokens','committed_usd','completed_usd']
        return {k:(v or 0) for k,v in zip(fields,row)}


def interruption(directory):
    log = directory/'model_events.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            entry = json.loads(line)
            if entry['event']=='request_error':
                return entry.get('reason', 'provider_error')
            if entry['event']=='budget_stop' and entry['reason']=='dollar_cap_reservation':
                return entry['reason']
    return None


class MeteredChat(ChatProvider):
    def __init__(self, ledger, directory, **kwargs):
        super().__init__(**kwargs)
        self.ledger, self.directory = ledger, Path(directory)

    def send_messages(self, messages, stage, seed, json_output=False, max_tokens=None):
        stopped = interruption(self.directory)
        if stopped:
            raise RuntimeError(stopped)
        reserved = len(json.dumps(messages).encode())+256
        output_cap = self.max_output_tokens if max_tokens is None else min(max_tokens,self.max_output_tokens)
        log = self.directory/'model_events.jsonl'
        try:
            number = self.ledger.reserve(stage,reserved,output_cap)
        except RuntimeError as exc:
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
