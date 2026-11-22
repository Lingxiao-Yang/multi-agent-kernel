import shutil

from kernel_study import records


def test_hashes_follow_nested_search_modules_and_ignore_reporting(tmp_path, monkeypatch):
    for name in ('kernel_study', 'configs', 'knowledge'):
        shutil.copytree(records.ROOT/name, tmp_path/name)
    monkeypatch.setattr(records, 'ROOT', tmp_path)
    before = records.code_hashes()
    reporting = tmp_path/'kernel_study/campaign/report.py'
    reporting.write_text('# changed reporting\n')
    assert records.code_hashes()==before
    added = 'kernel_study/agents/strategies/example.py'
    path = tmp_path/added
    path.parent.mkdir()
    path.write_text('value = 1\n')
    assert records.code_hashes()=={**before, added:records.digest(path.read_text())}
    path.write_text('value = 2\n')
    assert records.code_hashes()[added]==records.digest(path.read_text())
