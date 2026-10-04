"""공시 후 진입 시점 연구. 실제 주문 없이 공개 데이터와 계산 근거를 저장한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STRATEGIES = {
    "next_open": (0,),
    "wait_5": (5,),
    "wait_10": (10,),
    "wait_20": (20,),
    "split_0_5_20": (0, 5, 20),
    "dip_5pct_within_20": (),
}


class Fetcher:
    def __init__(self, root: Path):
        self.root = root
        self.attempts: list[dict] = []
        self.client = httpx.Client(
            timeout=30, follow_redirects=True,
            headers={"User-Agent": os.getenv("SEC_USER_AGENT", "InvestingResearcher/0.1 personal research")},
        )

    def get(self, url: str, *, refresh: bool = False) -> dict:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.root / "cache" / (key + ".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not refresh:
            envelope = json.loads(path.read_text())
            self.attempts.append({"url": url, "status": "CACHE", "retrieved_at": envelope["retrieved_at"]})
            return envelope["data"]
        for attempt in range(4):
            try:
                time.sleep(0.4)
                response = self.client.get(url)
                response.raise_for_status()
                data = response.json()
                timestamp = datetime.now(timezone.utc).isoformat()
                path.write_text(json.dumps({"url": url, "retrieved_at": timestamp, "data": data}))
                self.attempts.append({"url": url, "status": "OK", "retry_count": attempt, "retrieved_at": timestamp})
                return data
            except (httpx.HTTPError, ValueError) as exc:
                self.attempts.append({"url": url, "status": "FAILED", "retry_count": attempt, "error": str(exc)})
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                if status in (400, 401, 403, 404) or attempt == 3:
                    raise
                time.sleep((2, 5, 15)[attempt])
        raise RuntimeError("request failed")


def normalize_holdings(data: list[list]) -> pd.DataFrame:
    columns = ["symbol", "issuer", "class", "cusip", "provider_value_thousands_usd",
               "provider_total_weight_pct", "shares", "principal", "option"]
    frame = pd.DataFrame(data, columns=columns)
    frame["provider_value_thousands_usd"] = pd.to_numeric(frame["provider_value_thousands_usd"], errors="raise")
    if frame.empty or (frame["provider_value_thousands_usd"] < 0).any():
        raise ValueError("EMPTY_OR_INVALID_HOLDINGS")
    # 분모도 현물만 포함한다. 제공자의 옵션 포함 비중을 재사용하지 않는다.
    spot = frame.loc[frame.option.fillna("").eq("") & frame.shares.notna() & frame.principal.isna()].copy()
    spot["shares"] = pd.to_numeric(spot["shares"], errors="raise")
    spot = spot.groupby(["cusip", "class"], dropna=False, as_index=False).agg(
        symbol=("symbol", "first"), issuer=("issuer", "first"),
        shares=("shares", "sum"), provider_value_thousands_usd=("provider_value_thousands_usd", "sum"),
    )
    total = spot.provider_value_thousands_usd.sum()
    if total <= 0 or (spot.shares <= 0).any():
        raise ValueError("INVALID_SPOT_TOTAL_OR_SHARES")
    spot["weight_pct"] = spot.provider_value_thousands_usd / total * 100
    spot["provider_symbol"] = spot.symbol
    overrides = json.loads((ROOT / "ticker_overrides.json").read_text())
    spot["mapping_source"] = "13f.info"
    for cusip, override in overrides.items():
        mask = spot.cusip.eq(cusip)
        if override.get("expected_class") and not spot.loc[mask, "class"].eq(override["expected_class"]).all():
            raise ValueError("OVERRIDE_SECURITY_CLASS_MISMATCH:" + cusip)
        spot.loc[mask, "symbol"] = override["symbol"]
        spot.loc[mask, "mapping_source"] = override["source"]
    return spot.sort_values("weight_pct", ascending=False)


def load_filings(fetch: Fetcher, cfg: dict, asof: str) -> pd.DataFrame:
    url = f'https://data.sec.gov/submissions/CIK{cfg["cik"]}.json'
    data = fetch.get(url, refresh=True)
    frame = pd.DataFrame(data["filings"]["recent"])
    frame = frame.loc[frame.form.isin(["13F-HR", "13F-HR/A"]) & (frame.filingDate <= asof)]
    originals = frame.loc[frame.form.eq("13F-HR")].sort_values("reportDate").tail(cfg["quarters"])
    amended = set(frame.loc[frame.form.eq("13F-HR/A"), "reportDate"])
    originals = originals.copy()
    originals["amendment_present"] = originals.reportDate.isin(amended)
    return originals


def yahoo_prices(fetch: Fetcher, symbol: str, start: str, asof: str) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int((pd.Timestamp(asof, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    symbol = symbol.replace(".", "-") if not symbol.endswith((".TO", ".L", ".HK")) else symbol
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?period1={start_ts}&period2={end_ts}&interval=1d&events=div%2Csplits")
    result = fetch.get(url)["chart"]["result"][0]
    if result["meta"].get("currency") != "USD":
        raise ValueError("NON_USD_PRICE_REQUIRES_FX_CONVERSION")
    quote = result["indicators"]["quote"][0]
    frame = pd.DataFrame(quote)
    frame["adj_close"] = result["indicators"]["adjclose"][0]["adjclose"]
    frame.index = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    frame = frame.loc[~frame.index.duplicated()].sort_index()
    frame = frame.loc[frame.index <= pd.Timestamp(asof)].copy()
    factor = frame.adj_close / frame.close
    frame["adj_open"] = frame.open * factor
    if ((frame[["adj_open", "adj_close"]] <= 0).any()).any():
        raise ValueError("NONPOSITIVE_PRICE")
    # 기준일 당일의 장중 가격은 완성된 일봉으로 사용하지 않는다.
    now = pd.Timestamp.now(tz="America/New_York")
    if now.hour < 17:
        frame = frame.loc[frame.index < now.tz_localize(None).normalize()]
    frame.attrs["provider_name"] = result["meta"].get("longName", result["meta"].get("shortName"))
    frame.attrs["currency"] = result["meta"].get("currency")
    frame.attrs["exchange"] = result["meta"].get("exchangeName")
    return frame


def first_session(calendar: pd.DatetimeIndex, accepted: str) -> int:
    local_date = pd.Timestamp(accepted).tz_convert("America/New_York").tz_localize(None).normalize()
    # 공개 전 종가/시가를 쓰지 않도록 공시 현지 날짜 다음 거래일 시가부터 비교한다.
    return int(calendar.searchsorted(local_date, side="right"))


def simulate(frame: pd.DataFrame, calendar: pd.DatetimeIndex, base: int,
             horizon: int, strategy: str, cost: float) -> dict:
    dates = calendar[base:base + horizon + 1]
    if len(dates) != horizon + 1:
        raise ValueError("IMMATURE_HORIZON")
    prices = frame.reindex(dates)
    if prices[["adj_open", "adj_close"]].isna().any().any():
        raise ValueError("MISSING_PRICE_OR_DELISTING_REQUIRES_REVIEW")
    offsets = STRATEGIES[strategy]
    if strategy == "dip_5pct_within_20":
        anchor = prices.adj_open.iloc[0]
        signals = np.flatnonzero(prices.adj_close.iloc[:20].to_numpy() <= anchor * .95)
        offsets = (int(signals[0]) + 1,) if len(signals) else ()
    wealth = np.ones(len(dates))
    invested = 0.0
    for offset in offsets:
        allocation = 1 / len(offsets)
        units = allocation / (prices.adj_open.iloc[offset] * (1 + cost))
        wealth[offset:] += units * prices.adj_close.iloc[offset:].to_numpy() - allocation
        invested += units * prices.adj_close.iloc[-1]
    wealth[-1] -= invested * cost
    peak = np.maximum.accumulate(np.r_[1.0, wealth])
    drawdown = (np.r_[1.0, wealth] / peak - 1).min()
    return {
        "return_pct": float((wealth[-1] - 1) * 100),
        "max_drawdown_pct": float(drawdown * 100),
        "filled": bool(offsets),
        "first_entry_date": str(dates[offsets[0]].date()) if offsets else None,
        "exit_date": str(dates[-1].date()),
    }


def summarize(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon, strategy, subset), part in events.groupby(["horizon", "strategy", "subset"]):
        # 동일 분기 여러 종목의 높은 상관을 숨기지 않도록 분기별 평균을 먼저 낸다.
        quarter = part.groupby("report_date")[["return_pct", "excess_vs_next_open_pp", "excess_vs_spy_pp"]].mean()
        rng = np.random.default_rng(20260921)
        excess = quarter.excess_vs_next_open_pp.to_numpy()
        bootstrap = rng.choice(excess, (3000, len(excess)), replace=True).mean(axis=1)
        rows.append({
            "horizon": horizon, "strategy": strategy, "subset": subset,
            "events": len(part), "quarters": len(quarter),
            "mean_return_pct": quarter.return_pct.mean(),
            "median_event_return_pct": part.return_pct.median(),
            "mean_excess_vs_next_open_pp": excess.mean(),
            "quarter_bootstrap_low_pp": np.quantile(bootstrap, .025),
            "quarter_bootstrap_high_pp": np.quantile(bootstrap, .975),
            "mean_excess_vs_spy_pp": quarter.excess_vs_spy_pp.mean(),
            "positive_return_rate_pct": part.return_pct.gt(0).mean() * 100,
            "mean_event_drawdown_pct": part.max_drawdown_pct.mean(),
            "fill_rate_pct": part.filled.mean() * 100,
        })
    return pd.DataFrame(rows)


def write_note(output: Path, watch: pd.DataFrame, summary: pd.DataFrame, metadata: dict) -> None:
    lines = [
        "# 드러켄밀러 5% 이상 보유종목 및 공시 후 진입 시점 연구", "",
        f"조사 기준일: {metadata['asof']}. 실제 가격 마지막 날짜: {metadata['latest_price_date']}.",
        "주식·ADR·ETF 현물 평가액 합계를 분모로 사용한다. 옵션과 원금 표시 채권은 제외한다.",
        "이 문서는 매수 후보 및 탐색적 사건 연구이며 실주문이나 운용 성과가 아니다.", "",
        "## 매수계획 후보", "",
        "| 종목 | 회사 | 현물 비중 | 보유 기준일 | 공시일 | 상태 |",
        "|---|---|---:|---|---|---|",
    ]
    for r in watch.to_dict("records"):
        lines.append(f"| {r['symbol']} | {r['issuer']} | {r['weight_pct']:.2f}% | {r['report_date']} | {r['filing_date']} | 검토 후보 |")
    lines += ["", "## 진입 시점 비교", "",
              "대기일은 거래일 수이다. next_open은 공시일 다음 거래일 시가, wait_5는 그 시점에서 5거래일 뒤 시가이다.",
              "종료일은 모든 전략에 동일하다. 대기자금 수익률은 0%, 매수·매도 비용은 각각 0.10%로 가정한다.",
              "수익률은 USD 배당·분할 조정 가격 기준이며 환율·세금은 반영하지 않았다.",
              "dip은 기준 시가보다 종가가 5% 낮아진 것을 확인한 다음 거래일 시가에 매수한다. 20거래일 안에 신호가 없으면 현금을 유지한다.",
              "분할매수는 next_open, +5, +20거래일에 각각 초기자금의 1/3을 투입한다.", "",
              "| 보유 거래일 | 구간 | 진입 방식 | 사건/분기 수 | 평균 수익률 | 즉시매수 대비 | 분기 부트스트랩 95% 구간 | 체결률 |",
              "|---:|---|---|---:|---:|---:|---|---:|"]
    for r in summary.to_dict("records"):
        lines.append(f"| {r['horizon']} | {r['subset']} | {r['strategy']} | {r['events']}/{r['quarters']} | {r['mean_return_pct']:.2f}% | {r['mean_excess_vs_next_open_pp']:+.2f}%p | [{r['quarter_bootstrap_low_pp']:+.2f}, {r['quarter_bootstrap_high_pp']:+.2f}] | {r['fill_rate_pct']:.1f}% |")
    lines += ["", "## 해석 범위와 누락", "",
              "- all은 전체, 2024년 이전 공시 보유분은 early, 이후는 later이다. 사전에 등록한 검증이 아닌 사후 탐색 구간이다.",
              "- 63/126/252/378거래일은 약 3/6/12/18개월이며 달력 개월과 정확히 같지 않다.",
              "- 동일 종목이 여러 분기 반복 포함되고 보유기간도 겹친다. 사건 수를 독립 표본 수로 해석하면 안 된다.",
              "- 분기 부트스트랩은 분기 간 중첩까지 제거하지 못한다. 신뢰구간은 탐색용이며 유의성 보장이 아니다.",
              "- 당시 공시별 5% 이상 현물 종목 전체를 대상으로 한다. 신규로 5%를 돌파한 종목만의 전략은 아니다.",
              "- 다음 공시에 비중이 줄어도 정해진 종료일까지 보유하는 진입 시점 연구이다. 매도 규칙·실계좌 전략 검증은 별도이다.",
              "- 최신 후보만 과거로 소급하지 않는다. 정정 공시가 있는 분기는 이번 연구에서 제외하고 기록한다.",
              "- 가격이 없는 종목이나 매핑 실패는 제외 내역을 남긴다. 상장폐지 수익률을 임의의 0으로 채우지 않는다.",
              "- SEC 제출 메타데이터를 사용한다. 원문 다운로드 차단 시 13f.info 공개 보유표를 fallback으로 사용한다.",
              "- fallback의 금액은 원천 표시 단위를 보존한다. SEC 표기와 금액 단위가 의심스러운 경우 절대 AUM을 확정하지 않는다.",
              "- Yahoo 현재 심볼 매핑과 소급 조정 가격을 사용하므로 완전한 과거 시점 데이터베이스는 아니다.",
              "- 일봉으로 비교하므로 공시 직후 분 단위 매수의 우열은 판단할 수 없다.",
              f"- 누락 및 미성숙 사건: {metadata['excluded_count']}건. 자세한 내용은 exclusions.csv.",
              "", "파일: watchlist.csv, historical_candidates.csv, timing_events.csv, timing_summary.csv, exclusions.csv, source_attempts.json, metadata.json."]
    (output / "research.md").write_text("\n".join(lines), encoding="utf-8")


def run(asof: str) -> Path:
    cfg = json.loads((ROOT / "config.json").read_text())
    output = ROOT / "reports" / asof
    output.mkdir(parents=True, exist_ok=True)
    fetch = Fetcher(ROOT / "data")
    excluded: list[dict] = []
    frames = []
    try:
        filings = load_filings(fetch, cfg, asof)
        if filings.empty:
            raise RuntimeError("NO_FILINGS")
        latest_requested = filings.reportDate.max()
        # SEC 원문 접근 상태를 기록하고, 차단 시 공개 보유표의 명시적 fallback으로 이동한다.
        latest = filings.iloc[-1]
        archive = f'https://www.sec.gov/Archives/edgar/data/{int(cfg["cik"])}/{latest.accessionNumber.replace("-", "")}/index.json'
        try:
            fetch.get(archive)
        except (httpx.HTTPError, ValueError) as exc:
            excluded.append({"reason": "SEC_ARCHIVE_UNAVAILABLE_PUBLIC_FALLBACK", "detail": str(exc)})
        for filing in filings.to_dict("records"):
            if filing["amendment_present"]:
                excluded.append({"report_date": filing["reportDate"], "reason": "AMENDMENT_REQUIRES_POINT_IN_TIME_RECONSTRUCTION"})
                continue
            accession = filing["accessionNumber"].replace("-", "")
            url = f"https://13f.info/data/13f/{accession}"
            try:
                holdings = normalize_holdings(fetch.get(url)["data"])
                selected = holdings.loc[holdings.weight_pct >= cfg["minimum_weight_pct"]].copy()
                selected["report_date"] = filing["reportDate"]
                selected["filing_date"] = filing["filingDate"]
                selected["accepted_at"] = filing["acceptanceDateTime"]
                selected["accession"] = filing["accessionNumber"]
                selected["holdings_source"] = url
                selected["data_quality"] = "FALLBACK_REQUIRES_UNIT_AND_MAPPING_REVIEW"
                frames.append(selected)
                print(f"holdings {filing['reportDate']}: {len(selected)} candidates", flush=True)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                excluded.append({"report_date": filing["reportDate"], "reason": "HOLDINGS_FAILED", "detail": str(exc)})
        if not frames:
            raise RuntimeError("DATA_COLLECTION_FAILED")
        candidates = pd.concat(frames, ignore_index=True)
        candidates.to_csv(output / "historical_candidates.csv", index=False)
        watch = candidates.loc[candidates.report_date.eq(latest_requested)].copy()
        watch["plan_status"] = "RESEARCH_CANDIDATE_NOT_ORDER"
        watch.to_csv(output / "watchlist.csv", index=False)
        start = str((pd.Timestamp(filings.reportDate.min()) - pd.Timedelta(days=10)).date())
        prices = {}
        for symbol in sorted(set(candidates.symbol.dropna()) | {"SPY"}):
            if not symbol:
                continue
            try:
                prices[symbol] = yahoo_prices(fetch, symbol, start, asof)
                print(f"prices {symbol}: {len(prices[symbol])} rows", flush=True)
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                excluded.append({"symbol": symbol, "reason": "PRICE_FAILED", "detail": str(exc)})
        pd.DataFrame([{"symbol": symbol, **price.attrs} for symbol, price in prices.items()]).to_csv(output / "price_identities.csv", index=False)
        if "SPY" not in prices or prices["SPY"].empty:
            raise RuntimeError("BENCHMARK_UNAVAILABLE")
        calendar = prices["SPY"].dropna(subset=["adj_open", "adj_close"]).index
        for index, row in watch.iterrows():
            if row.symbol not in prices:
                continue
            frame = prices[row.symbol]
            base = first_session(calendar, row.accepted_at)
            if base >= len(calendar) or calendar[base] not in frame.index:
                continue
            watch.loc[index, "first_post_filing_open_date"] = str(calendar[base].date())
            watch.loc[index, "latest_price_date"] = str(frame.index[-1].date())
            watch.loc[index, "latest_close_usd"] = frame.close.iloc[-1]
            watch.loc[index, "return_since_first_open_pct"] = (frame.adj_close.iloc[-1] / frame.loc[calendar[base], "adj_open"] - 1) * 100
        watch.to_csv(output / "watchlist.csv", index=False)
        events = []
        for row in candidates.to_dict("records"):
            symbol = row["symbol"]
            if symbol not in prices:
                excluded.append({"symbol": symbol, "report_date": row["report_date"], "reason": "NO_PRICE_MAPPING"})
                continue
            base = first_session(calendar, row["accepted_at"])
            for horizon in cfg["hold_sessions"]:
                try:
                    cost = cfg["cost_bps_per_side"] / 10000
                    results = {s: simulate(prices[symbol], calendar, base, horizon, s, cost) for s in STRATEGIES}
                    benchmark = simulate(prices["SPY"], calendar, base, horizon, "next_open", cost)
                    for strategy, result in results.items():
                        events.append({
                            "symbol": symbol, "report_date": row["report_date"], "accepted_at": row["accepted_at"],
                            "weight_pct": row["weight_pct"], "strategy": strategy, "horizon": horizon,
                            "subset": "early" if row["report_date"] < cfg["validation_from_report_date"] else "later",
                            "baseline_entry_date": str(calendar[base].date()), **result,
                            "excess_vs_next_open_pp": result["return_pct"] - results["next_open"]["return_pct"],
                            "excess_vs_spy_pp": result["return_pct"] - benchmark["return_pct"],
                        })
                except ValueError as exc:
                    excluded.append({"symbol": symbol, "report_date": row["report_date"], "horizon": horizon, "reason": str(exc)})
        if not events:
            raise RuntimeError("NO_COMPLETE_EVENTS")
        event_frame = pd.DataFrame(events)
        summary = summarize(pd.concat([event_frame, event_frame.assign(subset="all")], ignore_index=True))
        event_frame.to_csv(output / "timing_events.csv", index=False)
        summary.to_csv(output / "timing_summary.csv", index=False)
        metadata = {
            "asof": asof, "run_timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_data_used": False, "config": cfg,
            "filings_requested": len(filings), "filings_collected": len(frames),
            "latest_report_date": latest_requested, "latest_watchlist_available": not watch.empty,
            "candidate_events": len(candidates), "unique_symbols": int(candidates.symbol.nunique()),
            "prices_collected": len(prices), "latest_price_date": str(calendar[-1].date()),
            "excluded_count": len(excluded), "holdings_fallback": "13f.info", "prices_source": "Yahoo chart",
            "performance_type": "EXPLORATORY_EVENT_STUDY_NOT_PORTFOLIO_BACKTEST",
            "absolute_value_status": "PROVIDER_UNITS_NOT_INDEPENDENTLY_RECONCILED",
            "optimal_entry_established": False, "orders_sent": False,
        }
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        write_note(output, watch, summary, metadata)
        print(watch[["symbol", "weight_pct", "report_date"]].to_string(index=False))
        print(summary.to_string(index=False))
        return output
    finally:
        (output / "source_attempts.json").write_text(json.dumps(fetch.attempts, ensure_ascii=False, indent=2))
        pd.DataFrame(excluded).to_csv(output / "exclusions.csv", index=False)
        fetch.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="드러켄밀러 5% 이상 현물 보유 종목과 진입 시점 연구")
    parser.add_argument("--asof", required=True, help="YYYY-MM-DD 기준일")
    args = parser.parse_args()
    datetime.strptime(args.asof, "%Y-%m-%d")
    print(run(args.asof))
