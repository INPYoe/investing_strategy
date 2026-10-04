import numpy as np
import pandas as pd
import pytest

from guru_tracker.validation_study import event_return, issuer_top


def test_next_session_and_roundtrip_cost():
    dates = pd.bdate_range('2026-01-02', periods=5)
    frame = pd.DataFrame({'adj_open': [10.]*5, 'adj_close': [10.]*5}, index=dates)
    result = event_return({'A': frame}, dates, '2026-01-02T15:00:00Z', horizon=2, cost_bps=25)
    assert result['entry_date'] == '2026-01-05'
    assert result['exit_date'] == '2026-01-06'
    assert result['return_pct'] == pytest.approx((.9975/1.0025-1)*100)


def test_missing_member_cannot_be_silently_dropped():
    dates = pd.bdate_range('2026-01-02', periods=5)
    frame = pd.DataFrame({'adj_open': 10., 'adj_close': 10.}, index=dates)
    broken = frame.copy()
    broken.iloc[2, 1] = np.nan
    with pytest.raises(ValueError, match='INCOMPLETE_PRICES'):
        event_return({'A': frame, 'B': broken}, dates, '2026-01-02T15:00:00Z', horizon=3)


def test_immature_not_truncated():
    dates = pd.bdate_range('2026-01-02', periods=5)
    with pytest.raises(ValueError, match='IMMATURE'):
        event_return({'A': pd.DataFrame()}, dates, '2026-01-02T15:00:00Z', horizon=60)


def test_share_classes_one_issuer():
    frame = pd.DataFrame({'cusip': ['02079K107', '02079K305', 'X', 'Y'],
                          'symbol': ['GOOG', 'GOOGL', 'XXX', 'YYY'], 'weight_pct': [3., 4., 6., 5.]})
    result = issuer_top(frame, count=2)
    assert result.symbol.tolist() == ['GOOGL', 'XXX']
    assert result.issuer_weight_pct.tolist() == [7., 6.]


def test_duplicate_symbol_rejected():
    frame = pd.DataFrame({'cusip': ['X', 'Y'], 'symbol': ['A', 'A'], 'weight_pct': [10., 9.]})
    with pytest.raises(ValueError, match='AMBIGUOUS'):
        issuer_top(frame, count=2)
