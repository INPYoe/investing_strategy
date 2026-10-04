"""개별 집중도와 공동 매집을 결합하는 분기별 검토 엔진. 주문과 점수 없음."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fuse(holdings: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    columns = ['cusip', 'symbol', 'state', 'conviction_managers', 'support_managers',
               'positive_managers', 'reducing_managers', 'unverified_managers',
               'evidence', 'orders_sent']
    if holdings.empty:
        return pd.DataFrame(columns=columns)
    h = holdings.copy()
    s = signals.copy()
    for frame in (h, s):
        frame['cusip'] = frame.cusip.astype(str).str.zfill(9)
        if frame.duplicated(['manager_id', 'cusip']).any():
            raise ValueError('DUPLICATE_MANAGER_SECURITY')
    if h.report_date.nunique() != 1 or set(s.current_report_date) != set(h.report_date):
        raise ValueError('MIXED_REPORT_PERIODS')
    weights = pd.to_numeric(h.weight_pct, errors='raise')
    if not np.isfinite(weights).all() or not weights.between(0, 100, inclusive='right').all():
        raise ValueError('INVALID_WEIGHTS')
    h['weight_pct'] = weights
    h['rank'] = h.groupby('manager_id').weight_pct.rank(method='min', ascending=False)
    h['count'] = h.groupby('manager_id').cusip.transform('count')
    h['mean_weight'] = h.groupby('manager_id').weight_pct.transform('mean')
    # 비중 순위와 평균 대비 중요도를 동시에 요구해 작은 시험 보유를 배제한다.
    h['meaningful'] = (h['rank'] <= np.ceil(h['count'] * .25)) & (weights >= h.mean_weight)
    h['conviction'] = (h['rank'] <= np.ceil(h['count'] * .10)) & (weights >= h.mean_weight * 1.5)
    indexed = s.set_index(['manager_id', 'cusip'])
    rows = []
    for cusip, group in h.groupby('cusip'):
        support, conviction, positive, reducing, unknown = set(), set(), set(), set(), set()
        for row in group.itertuples():
            manager = row.manager_id
            if row.meaningful:
                support.add(manager)
            if row.conviction:
                conviction.add(manager)
            if (manager, cusip) not in indexed.index:
                unknown.add(manager)
                continue
            change = indexed.loc[(manager, cusip)]
            # 신규는 실제 매수시점 확정이 아닌 공개표 신규 출현으로 해석한다.
            if change.previous_shares == 0 and change.current_shares > 0:
                if row.meaningful:
                    positive.add(manager)
                continue
            verified = str(change.get('direction_quality', '')) == 'VERIFIED_CORPORATE_ACTIONS'
            factor = change.get('split_factor', np.nan)
            if not verified or not np.isfinite(float(factor)) or float(factor) <= 0:
                unknown.add(manager)
                continue
            share_ratio = change.current_shares / (change.previous_shares * float(factor))
            weight_delta = change.current_weight_pct - change.previous_weight_pct
            # 비중은 %p 변화와 수량 변화가 함께 확인될 때만 방향으로 사용한다.
            if row.meaningful and share_ratio >= 1.10 and weight_delta >= .25:
                positive.add(manager)
            if share_ratio <= .90 and weight_delta <= -.25:
                reducing.add(manager)
        if not support:
            continue
        if reducing and positive:
            state = 'CONFLICT_REVIEW'
        elif len(reducing) >= 2:
            state = 'DISTRIBUTION_REVIEW'
        elif conviction & positive and len(positive) >= 2:
            state = 'CONVICTION_AND_CONSENSUS'
        elif len(positive) >= 2:
            state = 'CONSENSUS_ACCUMULATION'
        elif conviction & positive:
            state = 'INDIVIDUAL_CONVICTION'
        elif len(support) >= 2:
            state = 'CONSENSUS_HOLDING_WATCH'
        else:
            state = 'INDIVIDUAL_HOLDING_WATCH'
        rows.append(dict(cusip=cusip, symbol=group.symbol.iloc[0], state=state,
                         conviction_managers='|'.join(sorted(conviction)),
                         support_managers='|'.join(sorted(support)),
                         positive_managers='|'.join(sorted(positive)),
                         reducing_managers='|'.join(sorted(reducing)),
                         unverified_managers='|'.join(sorted(unknown)),
                         evidence=f'support={len(support)};positive={len(positive)};reducing={len(reducing)};unverified={len(unknown)}',
                         orders_sent=False))
    return pd.DataFrame(rows, columns=columns)


def export(report: Path) -> Path:
    metadata = json.loads((report / 'metadata.json').read_text())
    if metadata.get('status') != 'COMPLETE' or metadata.get('sample_data_used') is not False:
        raise ValueError('SOURCE_REPORT_NOT_COMPLETE')
    result = fuse(pd.read_csv(report / 'current_holdings.csv', dtype={'cusip': str}),
                  pd.read_csv(report / 'signals.csv', dtype={'cusip': str}))
    path = report / 'integrated_candidates.csv'
    result.to_csv(path, index=False)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='보존된 실제 분기 공시로 통합 후보 재계산')
    parser.add_argument('--report', type=Path, required=True)
    print(export(parser.parse_args().report))
