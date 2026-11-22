import pytest

from kernel_study.core import digest, write_json


def test_replay_keeps_fallbacks_and_checks_archived_source(tmp_path):
    from kernel_study.replay import archived_entries
    path = tmp_path / 'repair-strong-3584057' / 'campaign' / 'layernorm_irregular_single_22'
    path.mkdir(parents=True)
    source = 'baseline'
    (path / 'baseline.py').write_text(source)
    config = {'launches': []}
    write_json(path / 'baseline.request.json', {'config': config})
    write_json(path / 'summary.json', {
        'manifest': {'case': {'id':'layernorm_irregular'}, 'baseline_hash': digest(source)},
        'final': {'candidate_id': 0, 'config': config, 'result': {'status': 'ok'}}, 'history': []})
    entries = list(archived_entries(tmp_path))
    assert [entry['role'] for entry in entries] == ['start', 'final']
    assert entries[1]['candidate_id'] == 0
    assert entries[0]['key'] == entries[1]['key']
    (path / 'baseline.py').write_text('changed')
    with pytest.raises(ValueError, match='hash'):
        list(archived_entries(tmp_path))


def test_noise_rule_requires_two_wins_and_uses_calibration_only():
    from kernel_study.replay import threshold_from_controls, confirmed
    calibration = [[1.2, 1.04], [0.99, 1.13], [1.02, 1.03]]
    threshold = threshold_from_controls(calibration, 1.01)
    assert threshold == 1.04
    assert not confirmed([1.08, 1.04], threshold)
    assert confirmed([1.06, 1.05], threshold)
