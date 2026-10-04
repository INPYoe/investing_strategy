import pandas as pd
import pytest
from guru_tracker.fusion import fuse


def fixture():
    h, s = [], []
    for manager in ['m1', 'm2']:
        for i in range(100):
            weight = 2.5 if i == 0 else 97.5 / 99
            h.append(dict(manager_id=manager, cusip=f'{i:09d}', symbol=f'S{i}',
                          weight_pct=weight, report_date='2026-06-30'))
            s.append(dict(manager_id=manager, cusip=f'{i:09d}', previous_shares=0 if i == 0 else 100,
                          current_shares=100, previous_weight_pct=0 if i == 0 else weight,
                          current_weight_pct=weight, current_report_date='2026-06-30'))
    return pd.DataFrame(h), pd.DataFrame(s)


def test_two_new_high_conviction_positions_below_three_percent_qualify():
    h, s = fixture()
    r = fuse(h, s).iloc[0]
    assert r.state == 'CONVICTION_AND_CONSENSUS'
    assert r.positive_managers == 'm1|m2'


def test_unverified_increases_are_not_accumulation():
    h, s = fixture()
    s['previous_shares'] = 50
    s['previous_weight_pct'] = s.current_weight_pct / 2
    s['direction_quality'] = 'NO_LARGE_RATIO_TRIGGER'
    assert fuse(h, s).iloc[0].state == 'CONSENSUS_HOLDING_WATCH'


def test_verified_reduction_overrides_agreement():
    h, s = fixture()
    mask = (s.manager_id == 'm2') & (s.cusip == '000000000')
    s.loc[mask, 'previous_shares'] = 200
    s.loc[mask, 'previous_weight_pct'] = 5
    s.loc[mask, 'direction_quality'] = 'VERIFIED_CORPORATE_ACTIONS'
    s.loc[mask, 'split_factor'] = 1
    assert fuse(h, s).iloc[0].state == 'CONFLICT_REVIEW'


def test_period_mismatch_is_rejected():
    h, s = fixture()
    h.loc[0, 'report_date'] = '2026-03-31'
    with pytest.raises(ValueError, match='MIXED_REPORT'):
        fuse(h, s)
