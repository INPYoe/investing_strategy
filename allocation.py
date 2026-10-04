"""결정론적 그룹/종목 상한 역변동성 배분. 주문 제출 기능 없음."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ENTRY_STATES = {'INDIVIDUAL_CONVICTION', 'CONSENSUS_ACCUMULATION', 'CONVICTION_AND_CONSENSUS'}


def capped_weights(scores: dict[str, float], caps: dict[str, float], total=1.) -> dict[str, float]:
    """상한에 걸린 몫을 고정하고 남은 예산을 원점수 비례로 재배분한다."""
    keys = sorted(scores)
    if not keys or set(keys) != set(caps) or not np.isfinite(total) or total <= 0:
        raise ValueError('INVALID_ALLOCATION_INPUT')
    if any(not np.isfinite(scores[k]) or scores[k] <= 0 or not np.isfinite(caps[k])
           or not 0 < caps[k] <= 1 for k in keys):
        raise ValueError('INVALID_SCORE_OR_CAP')
    if sum(caps.values()) < total - 1e-12:
        raise ValueError('INFEASIBLE_FULL_INVESTMENT')
    remaining, active, result = total, keys, {}
    while active:
        denominator = sum(scores[k] for k in active)
        proposed = {k: remaining * scores[k] / denominator for k in active}
        limited = [k for k in active if proposed[k] > caps[k] + 1e-14]
        if not limited:
            result.update(proposed)
            break
        for k in limited:
            result[k] = caps[k]
            remaining -= caps[k]
        active = [k for k in active if k not in limited]
    return {k: result[k] for k in keys}


def allocate(document: dict) -> dict:
    """동일 입력으로 동일 출력. 연구 모드는 주문 가능한 계획으로 승격하지 않는다."""
    result = {'algorithm': 'hierarchical_capped_inverse_vol_v1', 'orders_sent': False,
              'execution_enabled': False, 'targets': [], 'exclusions': []}
    try:
        policy = document['policy']
        research = policy['research_mode']
        if type(research) is not bool or document.get('sample_data_used') is not False:
            raise ValueError('UNKNOWN_MODE_OR_SAMPLE_DATA')
        end = pd.Timestamp(document['asof'])
        sessions = pd.DatetimeIndex(document['sessions'])
        if (end.tzinfo is not None or sessions.tz is not None or sessions.has_duplicates
                or not sessions.is_monotonic_increasing or len(sessions) < 2 or sessions[-1] > end
                or (end-sessions[-1]).days > 5):
            raise ValueError('INVALID_OR_STALE_SESSION_CALENDAR')
        window = policy['lookback_returns']
        minimum = policy['min_returns']
        if type(window) is not int or type(minimum) is not int or not 20 <= minimum <= window:
            raise ValueError('INVALID_LOOKBACK')
        ceiling = float(policy['max_position'])
        short_ceiling = float(policy['short_history_cap'])
        if not 0 < short_ceiling <= ceiling <= 1:
            raise ValueError('INVALID_POSITION_CAP')
        candidates = document['candidates']
        symbols = [c['symbol'] for c in candidates]
        issuers = [c['issuer_id'] for c in candidates]
        if (len(set(symbols)) != len(symbols) or len(set(issuers)) != len(issuers)
                or any(not s for s in symbols+issuers)):
            raise ValueError('DUPLICATE_OR_MISSING_SECURITY_IDENTITY')
        scores, caps, groups, statistics = {}, {}, {}, {}
        for candidate in sorted(candidates, key=lambda c: c['symbol']):
            symbol = candidate['symbol']
            state = candidate['signal_state']
            allowed = state in ENTRY_STATES or (research and policy.get('include_consensus_holdings') is True
                                                 and state == 'CONSENSUS_HOLDING_WATCH')
            quality = candidate['data_quality']
            if not allowed or quality not in ({'VERIFIED', 'RESEARCH_ONLY'} if research else {'VERIFIED'}):
                result['exclusions'].append({'symbol': symbol, 'reason': 'SIGNAL_OR_DATA_NOT_ELIGIBLE'})
                continue
            if candidate.get('currency') != 'USD':
                raise ValueError('USD_PRICE_REQUIRED:' + symbol)
            raw = candidate['adjusted_closes']
            series = pd.Series(raw['values'], index=pd.DatetimeIndex(raw['dates']), dtype=float)
            if (series.index.tz is not None or series.index.has_duplicates
                    or not series.index.is_monotonic_increasing or (series.index > end).any()
                    or not np.isfinite(series).all() or (series <= 0).any()):
                raise ValueError('INVALID_PRICES:' + symbol)
            aligned = series.reindex(sessions).tail(window+1)
            # 상장 이전 선두 결측만 허용한다. 도중 결측 또는 마지막 날 부재는 거부한다.
            first = aligned.first_valid_index()
            aligned = aligned.loc[first:] if first is not None else aligned.iloc[:0]
            if aligned.isna().any() or len(aligned)-1 < minimum:
                result['exclusions'].append({'symbol': symbol, 'reason': 'INSUFFICIENT_OR_GAPPED_HISTORY'})
                continue
            returns = aligned.pct_change(fill_method=None).dropna()
            volatility = float(returns.std(ddof=1)*np.sqrt(252))
            if not np.isfinite(volatility) or volatility <= 1e-8:
                raise ValueError('INVALID_VOLATILITY:' + symbol)
            scores[symbol] = 1/volatility
            caps[symbol] = ceiling if len(returns) >= window else short_ceiling
            groups[symbol] = candidate['risk_group']
            statistics[symbol] = {'annualized_volatility': volatility, 'return_observations': len(returns),
                                  'signal_state': state, 'data_quality': quality, 'cap': caps[symbol]}
        if not scores:
            raise ValueError('NO_ELIGIBLE_CANDIDATES')
        group_scores, group_caps = {}, {}
        for group in sorted(set(groups.values())):
            members = [s for s in scores if groups[s] == group]
            declared = float(policy['group_caps'][group])
            if not 0 < declared <= 1:
                raise ValueError('INVALID_GROUP_CAP')
            group_scores[group] = sum(scores[s] for s in members)
            group_caps[group] = min(declared, sum(caps[s] for s in members))
        budgets = capped_weights(group_scores, group_caps)
        weights = {}
        for group, budget in budgets.items():
            members = [s for s in scores if groups[s] == group]
            weights.update(capped_weights({s:scores[s] for s in members}, {s:caps[s] for s in members}, budget))
        result.update(status='RESEARCH_REVIEW_REQUIRED' if research else 'TARGETS_COMPUTED_NOT_EXECUTION_APPROVAL',
                      asof=document['asof'], weights_sum=sum(weights.values()), group_weights=budgets,
                      targets=[{'symbol':s, 'target_weight':weights[s], 'risk_group':groups[s],
                                **statistics[s]} for s in sorted(weights)])
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        result.update(status='BLOCKED', reason=str(exc), targets=[])
    return result


def main():
    parser = argparse.ArgumentParser(description='결정론적 목표비중 연구. 주문 없음.')
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    result = allocate(json.loads(raw))
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(result['status'])
    return 2 if result['status'] == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
