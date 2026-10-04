"""캐시 기반 데이터 감사와 공시 후 동일비중 기준선. 주문/네트워크 호출 없음."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from guru_tracker.study import ROOT, Fetcher, normalize_holdings, first_session, yahoo_prices


def collect_inputs(asof: str, quarters: int, output: Path | None = None):
    """가격/보유표 fallback을 명시해 연구 캐시만 보충한다. 증권사에는 접속하지 않는다."""
    registry = json.loads((ROOT/'managers.json').read_text())
    fetch = Fetcher(ROOT/'data')
    symbols = {'SPY', 'QQQ'}
    errors, dates = [], []
    try:
        for manager in registry['production_research_cohort']:
            try:
                data = fetch.get(f'https://data.sec.gov/submissions/CIK{manager["cik"]}.json', refresh=True)
                filings = pd.DataFrame(data['filings']['recent'])
                stamps = pd.to_datetime(filings.acceptanceDateTime, utc=True)
                filings = filings.loc[stamps < (pd.Timestamp(asof)+pd.Timedelta(days=1)).tz_localize('America/New_York')]
                originals = filings.loc[filings.form.eq('13F-HR')].sort_values('reportDate').tail(quarters)
                for filing in originals.to_dict('records'):
                    try:
                        if ((filings.form.eq('13F-HR/A')) & filings.reportDate.eq(filing['reportDate'])).any():
                            continue
                        raw = fetch.get('https://13f.info/data/13f/'+filing['accessionNumber'].replace('-', ''))
                        holdings = issuer_top(normalize_holdings(raw['data']))
                        symbols.update(holdings.symbol)
                        dates.append(filing['reportDate'])
                    except Exception as exc:
                        errors.append({'manager': manager['id'], 'accession': filing['accessionNumber'],
                                       'error_type': type(exc).__name__})
            except Exception as exc:
                errors.append({'manager': manager['id'], 'error_type': type(exc).__name__})
        start = min(dates) if dates else asof
        for symbol in sorted(symbols):
            try:
                yahoo_prices(fetch, symbol, start, asof)
            except Exception as exc:
                errors.append({'symbol': symbol, 'error_type': type(exc).__name__})
    finally:
        out = output or ROOT/'reports'/asof/'validation'
        out.mkdir(parents=True, exist_ok=True)
        (out/'collection.json').write_text(json.dumps({'sample_data_used': False, 'orders_sent': False,
            'holdings_source': '13f.info_PUBLIC_FALLBACK', 'price_source': 'Yahoo_USD_ADJUSTED_FALLBACK',
            'errors': errors, 'attempts': fetch.attempts}, ensure_ascii=False, indent=2))
        fetch.client.close()


class Cache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.prices = {}
        self.used = {}
        for path in directory.glob('*.json'):
            if path.name.startswith('._'):
                continue
            try:
                doc = json.loads(path.read_text())
                if '/v8/finance/chart/' not in doc.get('url', ''):
                    continue
                raw = doc['data']['chart']['result'][0]
                if raw['meta'].get('currency') != 'USD':
                    continue
                frame = pd.DataFrame(raw['indicators']['quote'][0])
                frame['adj_close'] = raw['indicators']['adjclose'][0]['adjclose']
                frame['adj_open'] = frame.open * frame.adj_close / frame.close
                frame.index = pd.to_datetime(raw['timestamp'], unit='s', utc=True).tz_convert(
                    'America/New_York').normalize().tz_localize(None)
                if frame.index.duplicated().any():
                    continue
                symbol = raw['meta']['symbol']
                self.prices.setdefault(symbol, []).append((frame.sort_index(), path, doc))
            except (ValueError, KeyError, IndexError, TypeError, UnicodeError):
                continue

    def record(self, path, doc):
        self.used[str(path)] = {'url': doc['url'], 'retrieved_at': doc.get('retrieved_at'),
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    def get(self, url):
        path = self.directory / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        if not path.exists():
            raise ValueError('CACHE_MISS')
        doc = json.loads(path.read_text())
        if doc['url'] != url:
            raise ValueError('CACHE_IDENTITY_MISMATCH')
        self.record(path, doc)
        return doc['data']

    def prices_for(self, symbol, start, end):
        candidates = self.prices.get(symbol, [])
        for frame, path, doc in sorted(candidates, key=lambda x: x[2].get('retrieved_at', ''), reverse=True):
            if len(frame) and frame.index.min() <= start and frame.index.max() >= end:
                self.record(path, doc)
                return frame.loc[start:end]
        raise ValueError('PRICE_RANGE_MISSING:' + str(symbol))


def issuer_top(holdings, count=5):
    frame = holdings.copy()
    if frame.cusip.duplicated().any():
        raise ValueError('DUPLICATE_CUSIP')
    # 명시적으로 확인한 두 Alphabet 주식 종류만 합친다. CUSIP 접두사 추측은 하지 않는다.
    frame['issuer_key'] = frame.cusip.replace({'02079K107': 'ALPHABET', '02079K305': 'ALPHABET'})
    values = frame.groupby('issuer_key').weight_pct.sum()
    chosen = values.sort_values(ascending=False, kind='stable').head(count).index
    rows = []
    for key in chosen:
        group = frame.loc[frame.issuer_key.eq(key)].sort_values(['weight_pct', 'cusip'], ascending=[False, True])
        row = group.iloc[0].to_dict()
        row['issuer_weight_pct'] = values[key]
        rows.append(row)
    selected = pd.DataFrame(rows)
    if len(selected) != count or selected.symbol.isna().any() or selected.symbol.duplicated().any():
        raise ValueError('INCOMPLETE_OR_AMBIGUOUS_BASKET')
    return selected


def event_return(frames: dict, calendar, accepted_at, horizon=60, cost_bps=25):
    if horizon < 1 or not 0 <= cost_bps < 10000 or not frames:
        raise ValueError('INVALID_EVENT_SETTINGS')
    start = first_session(calendar, accepted_at)
    dates = calendar[start:start+horizon]
    if len(dates) != horizon:
        raise ValueError('IMMATURE_HORIZON')
    curves = []
    for symbol, frame in frames.items():
        prices = frame.reindex(dates)
        values = prices[['adj_open', 'adj_close']].to_numpy(float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('INCOMPLETE_PRICES:' + symbol)
        curves.append(prices.adj_close.to_numpy() / prices.adj_open.iloc[0])
    fee = cost_bps / 10000
    nav = np.mean(curves, axis=0) / (1+fee)
    nav[-1] *= 1-fee
    all_nav = np.r_[1., nav]
    return {'entry_date': str(dates[0].date()), 'exit_date': str(dates[-1].date()),
            'return_pct': (nav[-1]-1)*100,
            'max_drawdown_pct': (all_nav / np.maximum.accumulate(all_nav)-1).min()*100}


def run(asof: str, quarters=8, horizon=60, cost_bps=25, output: Path | None = None):
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is not None or quarters < 1:
        raise ValueError('INVALID_ASOF_OR_QUARTERS')
    cache = Cache(ROOT / 'data' / 'cache')
    output = output or ROOT / 'reports' / asof / 'validation'
    output.mkdir(parents=True, exist_ok=True)
    registry = json.loads((ROOT/'managers.json').read_text())
    events, audits = [], []
    selection_asof = registry['selection_asof']
    # 완성된 기준일 일봉만 사용한다. 캐시 수집 시점이 현재여도 과거로 신호를 이동하지 않는다.
    for manager in registry['production_research_cohort']:
        try:
            raw = cache.get(f'https://data.sec.gov/submissions/CIK{manager["cik"]}.json')
            filings = pd.DataFrame(raw['filings']['recent'])
            stamps = pd.to_datetime(filings.acceptanceDateTime, utc=True)
            filings = filings.loc[stamps < (cutoff + pd.Timedelta(days=1)).tz_localize('America/New_York')]
            originals = filings.loc[filings.form.eq('13F-HR')].sort_values('reportDate').tail(quarters)
            if originals.empty:
                raise ValueError('NO_ORIGINAL_FILINGS')
        except (ValueError, KeyError) as exc:
            audits.append({'manager': manager['id'], 'status': 'UNAVAILABLE', 'reason': str(exc)})
            continue
        for filing in originals.to_dict('records'):
            identity = {'manager': manager['id'], 'report_date': filing['reportDate'],
                        'accepted_at': filing['acceptanceDateTime'], 'accession': filing['accessionNumber']}
            try:
                if len(originals.loc[originals.reportDate.eq(filing['reportDate'])]) != 1:
                    raise ValueError('DUPLICATE_ORIGINAL')
                if ((filings.form.eq('13F-HR/A')) & (filings.reportDate.eq(filing['reportDate']))).any():
                    raise ValueError('AMENDMENT_RECONSTRUCTION_REQUIRED')
                url = 'https://13f.info/data/13f/' + filing['accessionNumber'].replace('-', '')
                holdings = normalize_holdings(cache.get(url)['data'])
                vals = holdings[['shares', 'weight_pct', 'provider_value_thousands_usd']].to_numpy(float)
                if not np.isfinite(vals).all() or (vals <= 0).any() or not np.isclose(holdings.weight_pct.sum(), 100):
                    raise ValueError('INVALID_HOLDINGS_NUMBERS')
                basket = issuer_top(holdings)
                audits.append({**identity, 'status': 'STRUCTURAL_PASS_SOURCE_REVIEW',
                    'holdings': len(holdings), 'weight_sum': holdings.weight_pct.sum(),
                    'official_xml_reconciled': False, 'corporate_actions_verified': False,
                    'valuation_unit_verified': False, 'basket': '|'.join(basket.symbol),
                    'reason': '공개 fallback 구조 검증만 통과. 원문 금액 단위·기업행위 검증 미완료'})
            except (ValueError, KeyError) as exc:
                audits.append({**identity, 'status': 'UNAVAILABLE', 'reason': str(exc)})
                events.append({**identity, 'status': 'EXCLUDED', 'reason': str(exc)})
                continue
            try:
                accepted_day = pd.Timestamp(filing['acceptanceDateTime']).tz_convert('America/New_York').tz_localize(None).normalize()
                # 기준일이 가격 캐시의 마지막 날보다 늦을 수 있어 마지막 실제 SPY 일봉을 상한으로 삼는다.
                choices = cache.prices.get('SPY', [])
                if not choices:
                    raise ValueError('BENCHMARK_CALENDAR_MISSING')
                spy, path, doc = max(choices, key=lambda x: len(x[0].loc[accepted_day:cutoff]))
                cache.record(path, doc)
                calendar = spy.loc[accepted_day:cutoff].index
                i = first_session(calendar, filing['acceptanceDateTime'])
                dates = calendar[i:i+horizon]
                if len(dates) != horizon:
                    raise ValueError('IMMATURE_HORIZON')
                prices = {s: cache.prices_for(s, dates[0], dates[-1]) for s in basket.symbol}
                result = event_return(prices, calendar, filing['acceptanceDateTime'], horizon, cost_bps)
                row = {**identity, **result, 'status': 'EXPLORATORY_VALID', 'basket': '|'.join(basket.symbol)}
                for benchmark in ('SPY', 'QQQ'):
                    frame = cache.prices_for(benchmark, dates[0], dates[-1])
                    r = event_return({benchmark: frame}, calendar, filing['acceptanceDateTime'], horizon, cost_bps)
                    row[benchmark+'_return_pct'] = r['return_pct']
                    row[benchmark+'_excess_pp'] = result['return_pct'] - r['return_pct']
                events.append(row)
            except (ValueError, KeyError) as exc:
                events.append({**identity, 'status': 'EXCLUDED', 'reason': str(exc)})
    event_frame = pd.DataFrame(events)
    valid = event_frame.loc[event_frame.status.eq('EXPLORATORY_VALID')] if len(event_frame) else pd.DataFrame()
    summaries = []
    for manager in registry['production_research_cohort']:
        group = valid.loc[valid.manager.eq(manager['id'])] if len(valid) else pd.DataFrame()
        summaries.append({'manager': manager['id'], 'valid_events': len(group),
                          'median_excess_spy_pp': group.SPY_excess_pp.median() if len(group) else None,
                          'median_excess_qqq_pp': group.QQQ_excess_pp.median() if len(group) else None,
                          'spy_win_rate': float(group.SPY_excess_pp.gt(0).mean()) if len(group) else None})
    pd.DataFrame(audits).to_csv(output/'data_audit.csv', index=False)
    event_frame.to_csv(output/'baseline_events.csv', index=False)
    pd.DataFrame(summaries).to_csv(output/'baseline_summary.csv', index=False)
    meta = {'asof': asof, 'run_timestamp': datetime.now(timezone.utc).isoformat(),
            'sample_data_used': False, 'orders_sent': False, 'network_used': False,
            'status': 'EXPLORATORY_ONLY' if len(valid) else 'INSUFFICIENT_DATA',
            'quarters_per_manager': quarters, 'horizon_sessions': horizon, 'cost_bps_each_side': cost_bps,
            'valid_events': len(valid), 'excluded_events': len(events)-len(valid),
            'cohort_selection_asof': selection_asof,
            'ticker_overrides': json.loads((ROOT/'ticker_overrides.json').read_text()),
            'ticker_overrides_sha256': hashlib.sha256((ROOT/'ticker_overrides.json').read_bytes()).hexdigest(),
            'limitations': ['현재 선택한 구루의 과거 성과이므로 구루 선택 생존편향 존재',
                '상위 5기업 동일비중 기준선이며 신규·증액 전략 또는 위험균형 전략이 아님',
                '개별 공시 이벤트가 겹칠 수 있으며 누적 계좌 CAGR로 합산할 수 없음',
                '분할/배당 조정가격 fallback, CUSIP 가격 식별 및 SEC XML 대조 미완료',
                '수정공시가 있는 분기는 보수적으로 제외; 누락·상장폐지 종목의 제외편향 가능',
                '비용은 가정이며 실제 환전·세금·슬리피지·정수주 제약 미반영',
                '홀드아웃 검증 아님. 실제 투자 우위 판정 금지'],
            'sources': list(cache.used.values())}
    (output/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    lines = ['# 1차 데이터 감사 및 공시 후 기준선', '',
             f'유효 이벤트 {len(valid)}개 / 제외 {len(events)-len(valid)}개. 실주문 없음.', '',
             '공시 미국 현지 날짜 다음 거래일 시가에 상위 5기업을 동일비중 매수하고,',
             f'{horizon}번째 거래일 종가에 청산한다. 매수·매도 각각 {cost_bps}bp 비용 가정.', '',
             '| 구루 | 유효 이벤트 | SPY 대비 초과수익 중앙값(pp) | QQQ 대비(pp) |',
             '|---|---:|---:|---:|']
    for row in summaries:
        a = f"{row['median_excess_spy_pp']:.2f}" if row['valid_events'] else 'N/A'
        b = f"{row['median_excess_qqq_pp']:.2f}" if row['valid_events'] else 'N/A'
        lines.append(f"| {row['manager']} | {row['valid_events']} | {a} | {b} |")
    lines += ['', '## 판정', '', '전략 채택 판단은 보류한다. 계산 가능한 이벤트만의 탐색적 결과다.',
              '', '## 남은 검증', *['- '+x for x in meta['limitations']]]
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='네트워크·주문 없는 캐시 기반 전략 감사')
    parser.add_argument('--asof', required=True)
    parser.add_argument('--quarters', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=60)
    parser.add_argument('--collect', action='store_true', help='실제 공개 자료로 캐시 보충. 주문 없음.')
    parser.add_argument('--output', type=Path, help='이전 결과를 보존할 별도 출력 폴더')
    args = parser.parse_args()
    if args.collect:
        collect_inputs(args.asof, args.quarters, args.output)
    print(run(args.asof, args.quarters, args.horizon, output=args.output))
