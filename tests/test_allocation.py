import numpy as np
import pandas as pd
import pytest

from guru_tracker.allocation import allocate, capped_weights


# ---------- capped_weights ----------

def test_capped_weights_uncapped_stays_proportional():
    result = capped_weights({'A': 1., 'B': 3.}, {'A': .9, 'B': .9})
    assert result['A'] == pytest.approx(.25)
    assert result['B'] == pytest.approx(.75)
    assert sum(result.values()) == pytest.approx(1.)


def test_capped_weights_redistributes_capped_share():
    # B가 상한(.4)에 걸리면 남은 예산은 A/C에만 원점수 비례로 재분배된다.
    result = capped_weights({'A': 1., 'B': 8., 'C': 1.}, {'A': .9, 'B': .4, 'C': .9})
    assert result['B'] == pytest.approx(.4)
    assert result['A'] == pytest.approx(.3)
    assert result['C'] == pytest.approx(.3)
    assert sum(result.values()) == pytest.approx(1.)


def test_capped_weights_infeasible_raises():
    with pytest.raises(ValueError, match='INFEASIBLE_FULL_INVESTMENT'):
        capped_weights({'A': 1., 'B': 1.}, {'A': .3, 'B': .3})


@pytest.mark.parametrize('scores,caps', [
    ({'A': 1.}, {'B': 1.}),          # 키 불일치
    ({'A': -1.}, {'A': 1.}),         # 음수 점수
    ({'A': 1.}, {'A': 0.}),          # 상한 0
    ({'A': float('nan')}, {'A': 1.}),  # 비유한 점수
])
def test_capped_weights_invalid_input_raises(scores, caps):
    with pytest.raises(ValueError):
        capped_weights(scores, caps)


# ---------- allocate ----------

def price_series(asof: str, days: int, *, daily_return: float, start: float = 100.) -> dict:
    dates = pd.bdate_range(end=asof, periods=days)
    values = [start]
    for _ in range(len(dates) - 1):
        values.append(values[-1] * (1 + daily_return if len(values) % 2 else 1 - daily_return))
    return {'dates': [str(d.date()) for d in dates], 'values': values}


def document(asof='2026-01-30', *, candidates=None, policy_overrides=None):
    sessions = pd.bdate_range(end=asof, periods=280)
    policy = {'research_mode': True, 'include_consensus_holdings': True, 'lookback_returns': 252,
              'min_returns': 60, 'max_position': .9, 'short_history_cap': .1,
              'group_caps': {'g1': 1., 'g2': 1.}}
    policy.update(policy_overrides or {})
    return {'asof': asof, 'sample_data_used': False, 'sessions': [str(d.date()) for d in sessions],
            'policy': policy, 'candidates': candidates or []}


def candidate(symbol, issuer_id, *, group='g1', state='INDIVIDUAL_CONVICTION',
             quality='VERIFIED', currency='USD', daily_return=.01, days=280):
    return {'symbol': symbol, 'issuer_id': issuer_id, 'risk_group': group, 'currency': currency,
            'signal_state': state, 'data_quality': quality,
            'adjusted_closes': price_series('2026-01-30', days, daily_return=daily_return)}


def test_allocate_research_mode_weights_sum_to_one_and_respect_caps():
    doc = document(candidates=[
        candidate('LOWVOL', 'A', daily_return=.001),
        candidate('HIVOL', 'B', daily_return=.03),
    ])
    result = allocate(doc)
    assert result['status'] == 'RESEARCH_REVIEW_REQUIRED'
    assert result['weights_sum'] == pytest.approx(1., abs=1e-9)
    weights = {t['symbol']: t['target_weight'] for t in result['targets']}
    # 변동성이 낮은 종목이 역변동성 배분에서 더 큰 비중을 받는다.
    assert weights['LOWVOL'] > weights['HIVOL']
    assert all(t['target_weight'] <= doc['policy']['max_position'] + 1e-9 for t in result['targets'])


def test_allocate_excludes_ineligible_signal_state():
    doc = document(candidates=[
        candidate('OK1', 'A'), candidate('OK2', 'B'),
        candidate('EXCLUDED', 'C', state='CONFLICT_REVIEW'),
    ])
    result = allocate(doc)
    assert {t['symbol'] for t in result['targets']} == {'OK1', 'OK2'}
    assert {e['symbol'] for e in result['exclusions']} == {'EXCLUDED'}


def test_allocate_excludes_insufficient_price_history():
    doc = document(candidates=[
        candidate('OK1', 'A'), candidate('OK2', 'B'),
        candidate('TOO_SHORT', 'C', days=30),  # min_returns=60 미만
    ])
    result = allocate(doc)
    assert {t['symbol'] for t in result['targets']} == {'OK1', 'OK2'}
    assert any(e['symbol'] == 'TOO_SHORT' and e['reason'] == 'INSUFFICIENT_OR_GAPPED_HISTORY'
               for e in result['exclusions'])


def test_allocate_short_history_gets_short_cap_not_full_cap():
    doc = document(candidates=[
        candidate('OK1', 'A'), candidate('OK2', 'B'),
        candidate('SHORT', 'C', days=100),  # >= min_returns(60) but < lookback_returns(252)
    ])
    result = allocate(doc)
    short = next(t for t in result['targets'] if t['symbol'] == 'SHORT')
    assert short['cap'] == pytest.approx(doc['policy']['short_history_cap'])


def test_allocate_is_deterministic_for_same_input():
    doc = document(candidates=[candidate('A', '1'), candidate('B', '2', group='g2', state='CONSENSUS_ACCUMULATION')])
    assert allocate(doc) == allocate(doc)


def test_allocate_blocked_on_invalid_policy_mode():
    doc = document(candidates=[candidate('A', '1')], policy_overrides={})
    doc['policy']['research_mode'] = 'yes'  # bool이어야 함
    result = allocate(doc)
    assert result['status'] == 'BLOCKED'
    assert result['targets'] == []


def test_allocate_blocked_on_stale_session_calendar():
    doc = document(candidates=[candidate('A', '1')])
    doc['sessions'] = doc['sessions'][:-10]  # asof보다 6일 이상 오래된 마지막 세션
    result = allocate(doc)
    assert result['status'] == 'BLOCKED'
    assert 'STALE_SESSION_CALENDAR' in result['reason'] or 'INVALID' in result['reason']


def test_allocate_rejects_non_usd_currency():
    doc = document(candidates=[candidate('A', '1', currency='KRW')])
    result = allocate(doc)
    assert result['status'] == 'BLOCKED'
    assert 'USD_PRICE_REQUIRED' in result['reason']
