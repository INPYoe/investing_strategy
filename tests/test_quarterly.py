import json
from datetime import date

from guru_tracker.quarterly import run, scheduled_window


def test_window():
    assert scheduled_window(date(2026, 11, 20))
    assert not scheduled_window(date(2026, 9, 22))
    assert not scheduled_window(date(2026, 11, 19))


def test_complete_only_once(tmp_path):
    report = tmp_path / 'report'
    report.mkdir()
    (report / 'metadata.json').write_text(json.dumps({
        'sample_data_used': False, 'status': 'COMPLETE', 'managers_requested': 4,
        'managers_collected': 4, 'managers_unavailable': 0}))
    selection = tmp_path / 'selection'
    selection.mkdir()
    (selection / 'metadata.json').write_text(json.dumps({'selected_count': 3, 'eligible_candidates': 5}))
    calls = []
    def collect(day):
        calls.append(('collect', day))
        return report
    def mock_selector(asof):
        calls.append(('select', asof))
        return selection
    state = tmp_path / 'state'
    first = run(date(2026, 11, 20), state_dir=state, collector=collect, selector=mock_selector)
    second = run(date(2026, 11, 21), state_dir=state, collector=collect, selector=mock_selector)
    assert first['status'] == 'RESEARCH_COMPLETE_REVIEW_REQUIRED'
    assert second['status'] == 'ALREADY_RESEARCHED'
    assert len([c for c in calls if c[0] == 'collect']) == 1
    assert len([c for c in calls if c[0] == 'select']) == 1
    assert first['selected_count'] == 3
    assert first['eligible_candidates'] == 5
    assert first['orders_sent'] is False


def test_partial_retries_and_sample_rejected(tmp_path):
    (tmp_path / 'metadata.json').write_text(json.dumps({
        'sample_data_used': False, 'status': 'COMPLETE', 'managers_requested': 5,
        'managers_collected': 4, 'managers_unavailable': 1}))
    for day in (20, 21):
        result = run(date(2026, 11, day), state_dir=tmp_path/'state', collector=lambda _: tmp_path)
        assert result['status'] == 'PARTIAL_RETRY_REQUIRED'
    (tmp_path / 'metadata.json').write_text('{"sample_data_used": true}')
    result = run(date(2026, 11, 22), state_dir=tmp_path/'state', collector=lambda _: tmp_path)
    assert result['status'] == 'DATA_COLLECTION_FAILED'


def test_outside_window_no_collection(tmp_path):
    def fail(_):
        raise AssertionError('must not collect')
    result = run(date(2026, 9, 22), state_dir=tmp_path, collector=fail)
    assert result['status'] == 'OUTSIDE_QUARTER_WINDOW'
