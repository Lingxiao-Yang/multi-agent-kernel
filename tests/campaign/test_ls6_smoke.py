import json
import os
import subprocess
from pathlib import Path


def test_batch_runs_only_native_single_after_key_free_preflight(tmp_path):
    script = Path(__file__).resolve().parents[2] / 'scripts/ls6_matvec_smoke.sbatch'
    (tmp_path / 'venv/bin').mkdir(parents=True)
    (tmp_path / 'venv/bin/activate').write_text('export PATH="$SLURM_SUBMIT_DIR/venv/bin:$PATH"\n')
    (tmp_path / 'cuda-module').write_text('cuda/test\n')
    (tmp_path / '.openai_env').write_text('OPENAI_API_KEY=test-only\n')
    stub = tmp_path / 'venv/bin/python'
    stub.write_text(
        '#!/usr/bin/env python3\n'
        'import json, os, sys\n'
        "with open(os.environ['TRACE'], 'a') as out:\n"
        "    print(json.dumps([sys.argv[1:], bool(os.getenv('OPENAI_API_KEY'))]), file=out)\n"
    )
    stub.chmod(0o700)
    trace = tmp_path / 'trace'
    env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), HOME=str(tmp_path), TRACE=str(trace),
               SLURM_JOB_ID='123', OPENAI_API_KEY='ignore-inherited-test-value')
    subprocess.run(['bash', '-c', 'module() { :; }; source "$1"', 'test', str(script)],
                   env=env, check=True)
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    preflight = [(argv[argv.index('preflight')+1], key) for argv, key in calls if 'preflight' in argv
                 and '--case' in argv]
    assert preflight == [('preflight', False), ('stress', False)]
    assert any(argv[0].endswith('/probe_counters.py') and not key for argv, key in calls)
    searches = [(argv, key) for argv, key in calls if 'run' in argv]
    assert [argv[argv.index('--job')+1] for argv, _ in searches] == ['0']
    for argv, key in searches:
        assert key and argv[argv.index('run')+1] == 'smoke'
        assert argv[argv.index('--model')+1] == 'gpt-4.1-2025-04-14'
        assert argv[argv.index('--endpoint')+1] == 'https://api.openai.com/v1'
        assert argv[argv.index('--output')+1] == str(tmp_path/'campaign-native')
