from kernel_study.cli import main
from kernel_study.records import read_json


def test_report_runs_without_a_gpu_or_model(tmp_path, capsys):
    main(['report', 'development', str(tmp_path)])
    assert capsys.readouterr().out.strip() == 'incomplete 0'
    report = read_json(tmp_path/'development-report.json')
    assert report['paired_searches'] == 0 and not report['complete']
