"""공시 이후 고정 자본 포트폴리오 연구. 주문 API와 연결하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from guru_tracker.study import Fetcher, ROOT, first_session, normalize_holdings, yahoo_prices


class ResearchFetcher(Fetcher):
    def __init__(self, *, offline: bool):
        super().__init__(ROOT / "data")
        self.offline = offline

    def get(self, url: str, *, refresh: bool = False) -> dict:
        if self.offline:
            key = hashlib.sha256(url.encode()).hexdigest()
            if not (self.root / "cache" / (key + ".json")).exists():
                raise RuntimeError("OFFLINE_CACHE_MISS:" + url)
            refresh = False
        return super().get(url, refresh=refresh)


def eligible_holdings(holdings: pd.DataFrame, previous: set[str], exit_below: float) -> pd.DataFrame:
    if not 0 <= exit_below <= 5:
        raise ValueError("INVALID_EXIT_THRESHOLD")
    entry = holdings.weight_pct.ge(5)
    retained = holdings.cusip.isin(previous) & holdings.weight_pct.gt(0)
    if exit_below:
        retained &= holdings.weight_pct.ge(exit_below)
    selected = holdings.loc[entry | retained].copy()
    if selected.cusip.duplicated().any() or selected.symbol.duplicated().any():
        raise ValueError("DUPLICATE_SECURITY_REQUIRES_REVIEW")
    if selected.symbol.isna().any() or selected.symbol.eq("").any():
        raise ValueError("UNMAPPED_SECURITY")
    return selected


def target_weights(selected: pd.DataFrame, allocation: str, cap: float) -> dict[str, float]:
    if allocation not in {"equal", "guru"} or not 0 < cap <= 1:
        raise ValueError("INVALID_ALLOCATION")
    if selected.empty:
        return {}
    raw = np.ones(len(selected)) if allocation == "equal" else selected.weight_pct.to_numpy(float)
    if not np.isfinite(raw).all() or (raw <= 0).any():
        raise ValueError("INVALID_WEIGHTS")
    # 상한 초과분을 다른 종목에 재분배하지 않는다. 최소 2%의 현금을 유지한다.
    weights = np.minimum(raw / raw.sum() * .98, cap)
    return dict(zip(selected.cusip, weights.tolist()))


def rebalance(values: dict[str, float], cash: float, weights: dict[str, float], cost: float):
    if not 0 <= cost < .1 or cash < -1e-8:
        raise ValueError("INVALID_COST_OR_CASH")
    if any(not np.isfinite(v) or v < 0 for v in [cash, *values.values(), *weights.values()]):
        raise ValueError("INVALID_PORTFOLIO")
    if sum(weights.values()) > 1 + 1e-12:
        raise ValueError("LEVERAGE_FORBIDDEN")
    nav = cash + sum(values.values())
    securities = set(values) | set(weights)
    # 매매 비용 차감 후 NAV에 목표 비중을 적용하므로 매수대금+비용이 자본을 넘지 않는다.
    low, high = 0., nav
    for _ in range(64):
        net = (low + high) / 2
        fee = cost * sum(abs(net * weights.get(s, 0) - values.get(s, 0)) for s in securities)
        if net + fee > nav:
            high = net
        else:
            low = net
    targets = {s: low * w for s, w in weights.items() if w > 0}
    traded = sum(abs(targets.get(s, 0) - values.get(s, 0)) for s in securities)
    fee = traded * cost
    remaining = nav - fee - sum(targets.values())
    if remaining < -1e-8:
        raise RuntimeError("NEGATIVE_CASH")
    return targets, max(0., remaining), fee, traded


def simulate_portfolio(events: list[dict], prices: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex,
                       *, allocation: str, cap: float, exit_below: float, cost: float, delay: int = 0):
    if not events or delay < 0:
        raise ValueError("NO_EVENTS_OR_INVALID_DELAY")
    scheduled = {}
    for event in events:
        index = first_session(calendar, event["accepted_at"]) + delay
        if index >= len(calendar):
            raise ValueError("FILING_ENTRY_NOT_YET_AVAILABLE")
        day = calendar[index]
        if day in scheduled:
            raise ValueError("MULTIPLE_FILINGS_SAME_EXECUTION_DAY")
        scheduled[day] = event
    first = min(scheduled)
    units, symbols, previous = {}, {}, set()
    cash, initial = 10000., 10000.
    rows, trades = [], []

    def price(cusip: str, date: pd.Timestamp, field: str) -> float:
        symbol = symbols[cusip]
        if symbol not in prices or date not in prices[symbol].index:
            raise ValueError(f"MISSING_PRICE:{symbol}:{date.date()}")
        value = float(prices[symbol].at[date, field])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"INVALID_PRICE:{symbol}:{date.date()}")
        return value

    for day in calendar[calendar >= first]:
        fee = traded = 0.
        if day in scheduled:
            event = scheduled[day]
            selected = eligible_holdings(event["holdings"], previous, exit_below)
            for row in selected.itertuples():
                if row.cusip in symbols and symbols[row.cusip] != row.symbol:
                    raise ValueError("SECURITY_MAPPING_CHANGED:" + row.cusip)
                symbols[row.cusip] = row.symbol
            weights = target_weights(selected, allocation, cap)
            current = {s: q * price(s, day, "adj_open") for s, q in units.items()}
            targets, cash, fee, traded = rebalance(current, cash, weights, cost)
            for s in sorted(set(current) | set(targets)):
                change = targets.get(s, 0) - current.get(s, 0)
                if abs(change) > 1e-8:
                    trades.append({"date": str(day.date()), "accession": event["accession"],
                                   "accepted_at": event["accepted_at"], "cusip": s, "symbol": symbols[s],
                                   "side": "BUY" if change > 0 else "SELL", "notional_usd": abs(change),
                                   "cost_usd": abs(change) * cost, "target_weight": weights.get(s, 0)})
            units = {s: value / price(s, day, "adj_open") for s, value in targets.items()}
            previous = set(selected.cusip)
        closing = {s: q * price(s, day, "adj_close") for s, q in units.items()}
        nav = cash + sum(closing.values())
        rows.append({"date": day, "nav": nav, "cash": cash, "positions": len(units),
                     "max_weight": max(closing.values(), default=0) / nav,
                     "traded_usd": traded, "cost_usd": fee})
    if not rows:
        raise ValueError("NO_PRICE_DAYS")
    # 비교 종료일에는 전량 청산 비용을 반영한다. 실제 매도를 뜻하지 않는다.
    final_cost = sum(closing.values()) * cost
    rows[-1]["nav"] -= final_cost
    rows[-1]["cost_usd"] += final_cost
    rows[-1]["traded_usd"] += sum(closing.values())
    frame = pd.DataFrame(rows).set_index("date")
    series = np.r_[initial, frame.nav.to_numpy()]
    drawdown = series / np.maximum.accumulate(series) - 1
    years = ((frame.index[-1] - frame.index[0]).days + 1) / 365.25
    summary = {"start": str(frame.index[0].date()), "end": str(frame.index[-1].date()),
               "total_return_pct": (series[-1] / initial - 1) * 100,
               "cagr_pct": ((series[-1] / initial) ** (1 / years) - 1) * 100,
               "max_drawdown_pct": drawdown.min() * 100, "cost_usd": frame.cost_usd.sum(),
               "turnover_initial_capital": frame.traded_usd.sum() / initial,
               "max_observed_weight_pct": frame.max_weight.max() * 100,
               "mean_cash_pct": (frame.cash / frame.nav).mean() * 100,
               "filings": len(events), "trade_count": len(trades)}
    return summary, frame, pd.DataFrame(trades)


def load_inputs(fetch: Fetcher, asof: str):
    cfg = json.loads((ROOT / "config.json").read_text())
    data = fetch.get(f'https://data.sec.gov/submissions/CIK{cfg["cik"]}.json')
    filings = pd.DataFrame(data["filings"]["recent"])
    cutoff = pd.Timestamp(asof, tz="America/New_York") + pd.Timedelta(days=1)
    filings = filings.loc[pd.to_datetime(filings.acceptanceDateTime, utc=True) < cutoff]
    originals = filings.loc[filings.form.eq("13F-HR")].sort_values("reportDate").tail(cfg["quarters"])
    amendments = filings.loc[filings.form.eq("13F-HR/A"), "reportDate"]
    if originals.reportDate.isin(amendments).any():
        raise ValueError("AMENDMENT_REQUIRES_POINT_IN_TIME_RECONSTRUCTION")
    if originals.empty or originals.reportDate.duplicated().any():
        raise ValueError("MISSING_OR_DUPLICATE_FILINGS")
    periods = pd.PeriodIndex(originals.reportDate, freq="Q")
    if (np.diff(periods.asi8) != 1).any():
        raise ValueError("NONCONSECUTIVE_FILINGS")
    events, symbols = [], {"SPY"}
    for row in originals.itertuples():
        raw = fetch.get(f'https://13f.info/data/13f/{row.accessionNumber.replace("-", "")}')
        holdings = normalize_holdings(raw["data"])
        selected = holdings.loc[holdings.weight_pct.ge(5)]
        if selected.symbol.isna().any() or selected.symbol.eq("").any():
            raise ValueError("UNMAPPED_CANDIDATE")
        symbols.update(selected.symbol)
        events.append({"accession": row.accessionNumber, "report_date": row.reportDate,
                       "accepted_at": row.acceptanceDateTime, "holdings": holdings})
    start = str((pd.Timestamp(originals.reportDate.min()) - pd.Timedelta(days=10)).date())
    prices = {s: yahoo_prices(fetch, s, start, asof) for s in sorted(symbols)}
    calendar = prices["SPY"].dropna(subset=["adj_open", "adj_close"]).index
    return events, prices, calendar


def run(asof: str, offline: bool = True):
    output = ROOT / "reports" / asof / "portfolio"
    output.mkdir(parents=True, exist_ok=True)
    fetch = ResearchFetcher(offline=offline)
    metadata = {"asof": asof, "run_timestamp": datetime.now(timezone.utc).isoformat(),
                "sample_data_used": False, "orders_sent": False, "status": "RUNNING",
                "holdings_source": "13f.info_PUBLIC_FALLBACK", "price_source": "Yahoo_USD_adjusted",
                "performance_type": "EXPLORATORY_FIXED_CAPITAL_BACKTEST_NOT_ACTUAL_ACCOUNT",
                "optimal_strategy_established": False, "offline": offline}
    try:
        events, prices, calendar = load_inputs(fetch, asof)
        split = "2024-01-01"
        summaries, curves, all_trades = [], [], []
        variants = [("equal35", "equal", .35), ("guru35", "guru", .35), ("guru_uncapped", "guru", 1.)]
        for subset in ("all", "early", "later"):
            selected_events = [e for e in events if subset == "all" or
                               (e["report_date"] < split if subset == "early" else e["report_date"] >= split)]
            selected_calendar = calendar
            if subset == "early":
                next_event = next(e for e in events if e["report_date"] >= split)
                boundary = first_session(calendar, next_event["accepted_at"])
                selected_calendar = calendar[:boundary]
            for code, allocation, cap in variants:
                for threshold in (5., 3., 0.):
                    for bps in (10, 30):
                        name = f"{code}_exit{threshold:g}_cost{bps}"
                        summary, curve, trades = simulate_portfolio(
                            selected_events, prices, selected_calendar, allocation=allocation,
                            cap=cap, exit_below=threshold, cost=bps / 10000)
                        summary.update(strategy=name, subset=subset, allocation=allocation, cap=cap,
                                       exit_below=threshold, cost_bps=bps)
                        summaries.append(summary)
                        curves.append(curve.reset_index().assign(strategy=name, subset=subset))
                        all_trades.append(trades.assign(strategy=name, subset=subset))
            for bps in (10, 30):
                benchmark = [{**selected_events[0], "holdings": pd.DataFrame(
                    [{"cusip": "SPY", "symbol": "SPY", "weight_pct": 100.}])}]
                summary, curve, trades = simulate_portfolio(
                    benchmark, prices, selected_calendar, allocation="equal", cap=1., exit_below=5, cost=bps / 10000)
                summary.update(strategy=f"SPY_98pct_cost{bps}", subset=subset, allocation="benchmark",
                               cap=1., exit_below=5., cost_bps=bps)
                summaries.append(summary)
                curves.append(curve.reset_index().assign(strategy=summary["strategy"], subset=subset))
        result = pd.DataFrame(summaries)
        result.to_csv(output / "comparison.csv", index=False)
        pd.concat(curves, ignore_index=True).to_csv(output / "equity_curves.csv", index=False)
        pd.concat(all_trades, ignore_index=True).to_csv(output / "paper_trades.csv", index=False)
        pd.concat([e["holdings"].assign(report_date=e["report_date"], accession=e["accession"],
                                       accepted_at=e["accepted_at"]) for e in events]).to_csv(output / "filings_holdings.csv", index=False)
        metadata.update(status="COMPLETE", filings=len(events), symbols=len(prices),
                        latest_price_date=str(calendar[-1].date()), strategies=len(result))
        lines = ["# 공시 추종 포트폴리오 비교", "", "실제 주문이 아닌 탐색적 고정 자본 백테스트다. 최고 수익이나 미래 성과를 보장하지 않는다.",
                 "", "## 계산 규칙", "",
                 "- 당시 공개된 현물 비중 5% 이상에서 신규 편입. 원문 대조 미완료 공개 fallback을 사용한다.",
                 "- 공시 접수 뉴욕 날짜 다음 거래일 시가에 리밸런싱. 전일 가격으로 미리 매수하지 않는다.",
                 "- exit5: 비중 5% 미만, exit3: 신규 편입 5%/보유 유지 3% 이상, exit0: 공개 현물 보유표에서 사라질 때 제외.",
                 "- exit0는 실제 전량 매도의 확증이 아니다. 비공개 처리·종목 재편입·기업행동을 따로 검증해야 한다.",
                 "- 유지 종목도 매 공시마다 동일가중 또는 공개 비중으로 재조정한다. 매도 조건과 보유 수량 고정은 다르다.",
                 "- equal35/guru35는 리밸런싱 시 계좌 NAV의 종목당 35% 상한. 이후 주가 변화로 상한을 넘을 수 있다.",
                 "- 목표 주식 비중 98%, 상한 초과분은 현금. 잔여 현금은 이자 0, 레버리지 없음.",
                 "- 초기 USD 10,000, 배당·분할 조정 단위의 소수 수량을 사용한다. 실계좌 정수 주식 수익률이 아니다.",
                 "- 편도 10bp와 30bp 비용을 비교하고 최종 청산 비용도 반영. 환율·세금·실제 체결 충격 미반영.",
                 "- early와 later는 공시 보유 기준일 2024-01-01로 분리해 각각 새 자본에서 시작한다. all은 연속 운용이다.",
                 "- 분리는 사후 탐색이며 진정한 사전 등록 검증이 아니다. 여러 규칙 비교에는 과적합 위험이 있다.",
                 "- 비교용 SPY는 98% 매수 후 보유. 조정 단위·가격 누락은 0 또는 전일 값으로 채우지 않고 연구 전체를 실패시킨다.",
                 "", "## 편도 10bp 결과", "",
                 "| 구간 | 규칙 | 누적수익률 | CAGR | 최대낙폭 | 최대 종목비중 | 평균 현금 |", "|---|---|---:|---:|---:|---:|---:|"]
        for r in result.loc[result.cost_bps.eq(10)].itertuples():
            lines.append(f"| {r.subset} | {r.strategy} | {r.total_return_pct:.2f}% | {r.cagr_pct:.2f}% | {r.max_drawdown_pct:.2f}% | {r.max_observed_weight_pct:.1f}% | {r.mean_cash_pct:.1f}% |")
        lines += ["", "## 운용 전 한계", "",
                  "SEC 접수 시각은 공식 메타데이터, 보유 내역은 13f.info fallback이다. 공시 단위·수정 공시·CUSIP 기업행동 검증이 남아 있다.",
                  "13F는 전체 자산, 공매도, 현금과 투자 의도를 모두 보여주지 않는다. 드러켄밀러가 모든 후보를 18개월 보유한다는 뜻도 아니다.",
                  "수익률은 매니저의 실제 수익률이 아니다. 최신 후보에 소급 투자하지 않았지만 현재 티커 및 현재 조정가격 사용의 한계가 있다.",
                  "실주문과 청산 예약은 활성화하지 않는다. 이 결과를 주문 승인이나 수익 보장으로 해석하지 않는다."]
        (output / "research.md").write_text("\n".join(lines), encoding="utf-8")
        print(result.loc[result.cost_bps.eq(10), ["subset", "strategy", "total_return_pct", "cagr_pct", "max_drawdown_pct"]].to_string(index=False))
        return output
    except Exception as exc:
        metadata.update(status="FAILED", reason=str(exc))
        raise
    finally:
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        (output / "source_attempts.json").write_text(json.dumps(fetch.attempts, ensure_ascii=False, indent=2))
        fetch.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="고정 자본 구루 포트폴리오 비교, 실주문 없음")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--online", action="store_true", help="누락 캐시의 실제 원자료 다운로드 허용")
    args = parser.parse_args()
    print(run(args.asof, offline=not args.online))
