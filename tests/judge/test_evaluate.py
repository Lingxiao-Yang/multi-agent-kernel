import sys
from types import SimpleNamespace

import pytest

from kernel_study.judge import evaluate as worker
from kernel_study.records import read_json, write_json
from kernel_study.tasks.starts import start


def test_common_referee_checks_modules_before_evaluation(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(worker, 'evaluate', lambda r: seen.append(r) or {'status':'ok'})
    req, out, module = (tmp_path/name for name in ('request.json','output.json','candidate.py'))
    monkeypatch.setattr(sys, 'argv', ['worker', str(req), str(out)])
    write_json(req, {'case':{'public_id':4},'source':str(module),'interface':'native'})
    module.write_text(start({'family':'matvec'}))
    worker.main()
    assert read_json(out)['status']=='ok'
    module.write_text('import torch\ndef kernel_function(x): return torch.sum(x)\n')
    worker.main()
    assert len(seen)==1 and read_json(out)['status']=='static_rejected'


def test_worker_file_error_reaches_the_search(tmp_path, monkeypatch):
    missing = tmp_path/'candidate.py'

    def invoke(command, **kwargs):
        monkeypatch.setattr(sys,'argv',command[2:])
        worker.main()
        return SimpleNamespace(returncode=0,stderr='')

    monkeypatch.setattr(worker.subprocess,'run',invoke)
    with pytest.raises(OSError,match='candidate.py'):
        worker.run_worker(dict(source=str(missing),protocol={'worker_timeout_s':10}),tmp_path,'candidate')
