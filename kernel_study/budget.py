"""Per-search limits and campaign-wide inference accounting."""
import json
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path


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
