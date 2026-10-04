"""최신·직전 13F를 비교해 다중 구루의 검토 신호를 생성하고 연속 보유 이력을 추적한다. 주문은 만들지 않는다."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from guru_tracker.consensus import build_consensus_candidates
from guru_tracker.fusion import fuse
from guru_tracker.signal_rules import position_signal
from guru_tracker.study import Fetcher, ROOT, normalize_holdings


def expected_report_date(asof: str) -> str:
    date = pd.Timestamp(asof).normalize()
    quarter_ends = pd.date_range(date - pd.DateOffset(years=2), date, freq="QE")
    eligible = [d for d in quarter_ends if d + pd.Timedelta(days=45) <= date]
    if not eligible:
        raise ValueError("NO_REPORT_PERIOD_DUE")
    return str(eligible[-1].date())


def holdings_for(fetch: Fetcher, filing: dict) -> pd.DataFrame:
    accession = filing["accessionNumber"].replace("-", "")
    data = fetch.get(f"https://13f.info/data/13f/{accession}")
    frame = normalize_holdings(data["data"])
    if frame.empty or frame.cusip.duplicated().any():
        raise ValueError("INVALID_OR_DUPLICATE_HOLDINGS")
    return frame


def collect_quarters(fetch: Fetcher, manager: dict, asof: str,
                     n_quarters: int = 5) -> list[tuple[dict, pd.DataFrame]]:
    """Returns list of (filing_dict, holdings_df) from oldest to newest.

    Only includes periods with a single original (no amendment) and fetchable holdings.
    Raises ValueError if the latest expected period is missing or fewer than 2 valid
    periods are available.
    """
    data = fetch.get(f'https://data.sec.gov/submissions/CIK{manager["cik"]}.json', refresh=True)
    filings = pd.DataFrame(data["filings"]["recent"])
    filings = filings.loc[filings.form.isin(["13F-HR", "13F-HR/A"]) & (filings.filingDate <= asof)].copy()
    originals = filings.loc[filings.form.eq("13F-HR")].sort_values("reportDate")
    if originals.empty:
        raise ValueError("NO_ORIGINAL_FILINGS")
    expected = expected_report_date(asof)
    if originals.reportDate.max() != expected:
        raise ValueError(f"LATEST_REQUIRED_PERIOD_MISSING:{expected}")
    amended = set(filings.loc[filings.form.eq("13F-HR/A"), "reportDate"])
    periods = originals.reportDate.drop_duplicates().tail(n_quarters).tolist()
    result = []
    for period in periods:
        if period in amended:
            continue  # amendment creates a gap in consecutive-count history
        rows = originals.loc[originals.reportDate.eq(period)].sort_values("acceptanceDateTime")
        if len(rows) != 1:
            continue
        filing = rows.iloc[0].to_dict()
        try:
            holdings = holdings_for(fetch, filing)
            result.append((filing, holdings))
        except Exception:
            continue  # holdings failure also creates a gap
    if len(result) < 2:
        raise ValueError("TWO_PERIODS_WITH_VALID_HOLDINGS_REQUIRED")
    if result[-1][0]["reportDate"] != expected:
        raise ValueError(f"LATEST_VALID_PERIOD_NOT_CURRENT:{expected}")
    return result


def get_split_factor(fetch: Fetcher, symbol: str,
                     prev_date: str, curr_date: str) -> float | None:
    """Cumulative split factor between two quarter-end report dates.

    Returns 1.0 if Yahoo reports no splits, None on any fetch or parse error.
    """
    if not symbol:
        return None
    try:
        sym = symbol.replace(".", "-") if not symbol.endswith((".TO", ".L", ".HK")) else symbol
        start_ts = int(pd.Timestamp(prev_date, tz="UTC").timestamp())
        end_ts = int((pd.Timestamp(curr_date, tz="UTC") + pd.Timedelta(days=5)).timestamp())
        url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
               f"?period1={start_ts}&period2={end_ts}&interval=1d&events=div%2Csplits")
        result = fetch.get(url)["chart"]["result"][0]
        splits = result.get("events", {}).get("splits", {})
        if not splits:
            return 1.0
        factor = 1.0
        for split in splits.values():
            factor *= float(split["numerator"]) / float(split["denominator"])
        return factor
    except Exception:
        return None


def _is_adjacent_quarter(earlier: str, later: str) -> bool:
    """True if two report dates are one quarter apart (60–120 calendar days)."""
    gap = (pd.Timestamp(later) - pd.Timestamp(earlier)).days
    return 60 <= gap <= 120


def consecutive_count(pairs: list[tuple[dict, pd.DataFrame]], cusip: str) -> int:
    """Consecutive valid quarters (no period gaps) from most recent where cusip appears."""
    if not pairs:
        return 0
    count = 0
    rev = list(reversed(pairs))
    for i, (filing, df) in enumerate(rev):
        if i > 0 and not _is_adjacent_quarter(filing["reportDate"], rev[i - 1][0]["reportDate"]):
            break  # gap in history: stop counting
        if cusip not in set(df.cusip):
            break
        count += 1
    return count


def compare_manager(manager: dict, previous_filing: dict, current_filing: dict,
                    previous: pd.DataFrame, current: pd.DataFrame, policy: dict,
                    split_factors: dict[str, float] | None = None) -> list[dict]:
    old = {r.cusip: r for r in previous.itertuples()}
    new = {r.cusip: r for r in current.itertuples()}
    rows = []
    for cusip in sorted(set(old) | set(new)):
        before, after = old.get(cusip), new.get(cusip)
        before_data = None if before is None else {"weight_pct": before.weight_pct, "shares": before.shares}
        after_data = None if after is None else {"weight_pct": after.weight_pct, "shares": after.shares}
        split_factor = None if split_factors is None else split_factors.get(cusip)
        if before is not None and after is not None:
            raw_share_ratio = float(after.shares) / float(before.shares)
            raw_weight_ratio = float(after.weight_pct) / float(before.weight_pct)
        if before is None and after is not None:
            holding_direction = "NEW"
            weight_ratio = None
            share_ratio = None
        elif before is not None and after is None:
            holding_direction = "MISSING_REVIEW"
            weight_ratio = 0.0
            share_ratio = 0.0
        elif split_factor is None:
            holding_direction = "CORPORATE_ACTION_REVIEW"
            weight_ratio = raw_weight_ratio
            share_ratio = raw_share_ratio
        else:
            weight_ratio = raw_weight_ratio
            share_ratio = raw_share_ratio
            if weight_ratio >= 1.05 and share_ratio >= 1.05:
                holding_direction = "ACCUMULATING"
            elif weight_ratio <= .95 and share_ratio <= .95:
                holding_direction = "REDUCING"
            else:
                holding_direction = "MIXED_OR_FLAT"
        signal = position_signal(
            manager_id=manager["id"], accession=current_filing["accessionNumber"], cusip=cusip,
            previous=before_data, current=after_data, comparison_verified=True,
            split_factor=split_factor, absence_reviewed=False,
            minimum_new_weight=policy["entry"]["new_position_min_weight_pct"],
        )
        instrument = after or before
        rows.append({
            "manager_id": manager["id"], "manager_name": manager["name"], "lead_name": manager["lead_name"],
            "previous_report_date": previous_filing["reportDate"], "current_report_date": current_filing["reportDate"],
            "accepted_at": current_filing["acceptanceDateTime"], "accession": current_filing["accessionNumber"],
            "symbol": instrument.symbol, "issuer": instrument.issuer, "cusip": cusip,
            "previous_weight_pct": 0. if before is None else before.weight_pct,
            "current_weight_pct": 0. if after is None else after.weight_pct,
            "previous_shares": 0. if before is None else before.shares,
            "current_shares": 0. if after is None else after.shares,
            "weight_ratio": weight_ratio, "share_ratio": share_ratio,
            "split_factor": split_factor,
            "holding_direction": holding_direction,
            "direction_quality": (
                "YAHOO_SPLITS_VERIFIED" if split_factor is not None and before is not None and after is not None
                else "CORPORATE_ACTIONS_UNVERIFIED" if split_factor is None and before is not None and after is not None
                else "PUBLIC_TABLE_APPEARANCE_OR_ABSENCE_ONLY"
            ),
            "signal": signal["signal"], "evidence": "|".join(signal["evidence"]),
            "actionable_for_tomorrow_plan": signal["signal"] == "NEW_ENTRY_CANDIDATE",
            "data_source": "SEC_METADATA_AND_13F_INFO_PUBLIC_FALLBACK",
            "orders_sent": False,
        })
    return rows


def build_consecutive_counts(history_per_manager: dict[str, list[tuple[dict, pd.DataFrame]]],
                              cohort: list[dict]) -> pd.DataFrame:
    """Consecutive holding counts and entry conditions per manager per CUSIP in current quarter."""
    M = len(cohort)
    rows = []
    for manager in cohort:
        mid = manager["id"]
        pairs = history_per_manager.get(mid, [])
        if not pairs:
            continue
        current_filing, current_holdings = pairs[-1]
        ranked = current_holdings.sort_values("weight_pct", ascending=False).reset_index(drop=True)
        for rank_idx, h in enumerate(ranked.itertuples(), start=1):
            count = consecutive_count(pairs, h.cusip)
            q_weight = float(h.weight_pct) >= 5.0
            q_top5 = rank_idx <= 5
            rows.append({
                "manager_id": mid,
                "current_report_date": current_filing["reportDate"],
                "cusip": h.cusip,
                "symbol": h.symbol,
                "issuer": h.issuer,
                "current_weight_pct": float(h.weight_pct),
                "rank_in_manager": rank_idx,
                "consecutive_quarters": count,
                "qualifies_weight": q_weight,
                "qualifies_top5": q_top5,
                "meets_entry_condition": (q_weight or q_top5) and count >= 5,
                "M": M,
            })
    return pd.DataFrame(rows)


def run(asof: str) -> Path:
    registry = json.loads((ROOT / "managers.json").read_text())
    policy = json.loads((ROOT / "multi_guru_policy.json").read_text())
    output = ROOT / "reports" / asof / "multi_guru"
    output.mkdir(parents=True, exist_ok=True)
    fetch = Fetcher(ROOT / "data")
    statuses: list[dict] = []
    signals: list[dict] = []
    holdings: list[dict] = []
    history_per_manager: dict[str, list[tuple[dict, pd.DataFrame]]] = {}
    cohort = registry["production_research_cohort"]
    try:
        for manager in cohort:
            try:
                pairs = collect_quarters(fetch, manager, asof, n_quarters=5)
                previous_filing, previous = pairs[-2]
                current_filing, current = pairs[-1]
                # Look up split factors from Yahoo for holdings present in both quarters
                common = set(previous.cusip) & set(current.cusip)
                split_factors: dict[str, float] = {}
                for cusip in common:
                    sym = current.loc[current.cusip.eq(cusip), "symbol"].iloc[0]
                    factor = get_split_factor(fetch, sym, previous_filing["reportDate"], current_filing["reportDate"])
                    if factor is not None:
                        split_factors[cusip] = factor
                rows = compare_manager(manager, previous_filing, current_filing,
                                       previous, current, policy, split_factors)
                signals.extend(rows)
                holdings.extend(
                    current.assign(manager_id=manager["id"],
                                   report_date=current_filing["reportDate"],
                                   accession=current_filing["accessionNumber"]).to_dict("records"))
                history_per_manager[manager["id"]] = pairs
                statuses.append({
                    "manager_id": manager["id"], "manager_name": manager["name"],
                    "status": "COLLECTED",
                    "previous_report_date": previous_filing["reportDate"],
                    "current_report_date": current_filing["reportDate"],
                    "current_accession": current_filing["accessionNumber"],
                    "quarters_collected": len(pairs),
                    "split_factors_resolved": len(split_factors),
                    "holdings": len(current), "error": "",
                })
            except Exception as exc:
                statuses.append({
                    "manager_id": manager["id"], "manager_name": manager["name"],
                    "status": "UNAVAILABLE", "quarters_collected": 0,
                    "split_factors_resolved": 0, "error": str(exc),
                })

        status_frame = pd.DataFrame(statuses)
        signal_frame = pd.DataFrame(signals)
        holdings_frame = pd.DataFrame(holdings)
        consecutive_frame = build_consecutive_counts(history_per_manager, cohort)

        # Full holding history across all collected quarters and managers
        history_rows: list[dict] = []
        for mid, pairs in history_per_manager.items():
            for filing, df in pairs:
                history_rows.extend(
                    df.assign(manager_id=mid,
                              report_date=filing["reportDate"],
                              accession=filing["accessionNumber"]).to_dict("records"))
        history_frame = pd.DataFrame(history_rows)

        consensus_frame = build_consensus_candidates(holdings_frame, signal_frame, policy)
        integrated = fuse(holdings_frame, signal_frame) if holdings else pd.DataFrame()

        status_frame.to_csv(output / "manager_status.csv", index=False)
        signal_frame.to_csv(output / "signals.csv", index=False)
        holdings_frame.to_csv(output / "current_holdings.csv", index=False)
        consensus_frame.to_csv(output / "consensus_candidates.csv", index=False)
        consecutive_frame.to_csv(output / "consecutive_counts.csv", index=False)
        history_frame.to_csv(output / "holding_history.csv", index=False)
        integrated.to_csv(output / "integrated_candidates.csv", index=False)

        collected = int(status_frame.status.eq("COLLECTED").sum())
        actionable = (signal_frame.loc[signal_frame.actionable_for_tomorrow_plan]
                      if not signal_frame.empty else signal_frame)
        entry_candidates = (consecutive_frame.loc[consecutive_frame.meets_entry_condition]
                            if not consecutive_frame.empty else consecutive_frame)

        metadata = {
            "run_timestamp": datetime.now(timezone.utc).isoformat(), "asof": asof,
            "sample_data_used": False, "orders_sent": False,
            "managers_requested": len(status_frame),
            "managers_collected": collected,
            "managers_unavailable": int(len(status_frame) - collected),
            "entry_condition_candidates": int(entry_candidates.cusip.nunique()) if not entry_candidates.empty else 0,
            "new_entry_candidates": int(len(actionable)),
            "consensus_core_candidates": int(consensus_frame.candidate_level.eq("CORE").sum()),
            "consensus_watch_candidates": int(consensus_frame.candidate_level.eq("WATCH").sum()),
            "signals_requiring_review": int(signal_frame.signal.eq("REVIEW").sum()) if not signal_frame.empty else 0,
            "source": "SEC submissions metadata + 13f.info public holdings fallback + Yahoo Finance splits",
            "status": "COMPLETE" if collected == len(status_frame) and collected > 0 else (
                "PARTIAL_RETRY_REQUIRED" if collected > 0 else "DATA_COLLECTION_FAILED"),
        }
        (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
        (output / "source_attempts.json").write_text(json.dumps(fetch.attempts, ensure_ascii=False, indent=2))

        if not collected:
            raise RuntimeError("DATA_COLLECTION_FAILED")

        lines = [
            "# 다중 구루 최신 공시 신호", "",
            f"기준일 {asof}, 수집 성공 {collected}/{len(status_frame)}개 운용사.",
            "", "## 5분기 연속 보유 편입 조건 충족 후보", "",
            "조건: 최소 한 운용사에서 5분기 연속 보유 AND 현재 분기 상위 5위 또는 비중 5% 이상.", "",
            "| 종목 | CUSIP | 운용사 | 연속 분기 | 현재 비중 | 상위5 | 비중5% |",
            "|---|---|---|---:|---:|:---:|:---:|",
        ]
        if not entry_candidates.empty:
            for r in entry_candidates.sort_values(["cusip", "manager_id"]).itertuples():
                lines.append(f"| {r.symbol} | {r.cusip} | {r.manager_id} | {r.consecutive_quarters}"
                             f" | {r.current_weight_pct:.2f}% | {r.qualifies_top5} | {r.qualifies_weight} |")
        else:
            lines.append("| — | — | — | — | — | — | — |")

        lines += [
            "", "## 최신 분기 신규 등장 후보 (전 분기 없음, 현재 3% 이상)", "",
            "| 구루 | 종목 | 현재 비중 | 신호 | 근거 |", "|---|---|---:|---|---|",
        ]
        for r in actionable.sort_values(["manager_id", "current_weight_pct"], ascending=[True, False]).itertuples():
            lines.append(f"| {r.manager_name} | {r.symbol} | {r.current_weight_pct:.2f}% | {r.signal} | {r.evidence} |")

        lines += [
            "", "## 다중 구루 합의 후보 (탐색 참고용)", "",
            "| 등급 | 종목 | 유효 구루 수 | 구루 | 중앙 비중 | 방향 확인 |",
            "|---|---|---:|---|---:|---|",
        ]
        for r in consensus_frame.itertuples():
            lines.append(f"| {r.candidate_level} | {r.symbol} | {r.meaningful_manager_count}"
                        f" | {r.managers} | {r.median_meaningful_weight_pct:.2f}% | {r.entry_status} |")

        (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
        with (output / "summary.md").open("a", encoding="utf-8") as handle:
            handle.write("\n\n## 통합 신호 검토 (탐색 참고용)\n\n")
            handle.write("| 종목 | 통합 상태 | 근거 |\n|---|---|---|\n")
            for r in integrated.itertuples():
                handle.write(f"| {r.symbol} | {r.state} | {r.evidence} |\n")

        return output
    finally:
        fetch.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다중 구루 13F 비교 및 연속 보유 이력 추적")
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()
    print(run(args.asof))
