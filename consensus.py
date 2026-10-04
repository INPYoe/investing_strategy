"""여러 구루가 의미 있는 비중으로 동시에 보유한 합의 후보를 만든다."""
from __future__ import annotations

import pandas as pd


def _representative(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return ""
    modes = values.mode()
    return modes.iloc[0] if not modes.empty else values.iloc[0]


def build_consensus_candidates(holdings: pd.DataFrame, signals: pd.DataFrame, policy: dict) -> pd.DataFrame:
    cfg = policy["consensus"]
    columns = ["cusip", "symbol", "issuer", "candidate_level", "manager_count",
               "meaningful_manager_count", "managers", "total_disclosed_weight_pct",
               "median_meaningful_weight_pct", "max_manager_weight_pct", "positive_signal_count",
               "reducing_manager_count", "direction_state",
               "entry_status", "evidence", "orders_sent"]
    if holdings.empty:
        return pd.DataFrame(columns=columns)
    required = {"cusip", "symbol", "issuer", "manager_id", "weight_pct"}
    if not required.issubset(holdings.columns):
        raise ValueError("CONSENSUS_HOLDINGS_COLUMNS_MISSING")
    frame = holdings.copy()
    frame["cusip"] = frame["cusip"].astype(str).str.zfill(9)
    frame["weight_pct"] = pd.to_numeric(frame["weight_pct"], errors="raise")
    if frame[["cusip", "manager_id"]].duplicated().any() or (frame.weight_pct <= 0).any():
        raise ValueError("CONSENSUS_HOLDINGS_INVALID")

    positive_by_cusip: dict[str, int] = {}
    reducing_by_cusip: dict[str, int] = {}
    if not signals.empty:
        direction = (signals["holding_direction"] if "holding_direction" in signals
                     else pd.Series("", index=signals.index))
        positive = signals.loc[
            signals.signal.isin(["NEW_ENTRY_CANDIDATE", "INCREASE_ENTRY_CANDIDATE"])
            | direction.eq("ACCUMULATING")].copy()
        reducing = signals.loc[direction.eq("REDUCING")].copy()
        if not positive.empty:
            positive["cusip"] = positive["cusip"].astype(str).str.zfill(9)
            positive_by_cusip = positive.groupby("cusip").manager_id.nunique().to_dict()
        if not reducing.empty:
            reducing["cusip"] = reducing["cusip"].astype(str).str.zfill(9)
            reducing_by_cusip = reducing.groupby("cusip").manager_id.nunique().to_dict()

    rows = []
    minimum_weight = float(cfg["meaningful_manager_weight_pct"])
    for cusip, group in frame.groupby("cusip", sort=True):
        meaningful = group.loc[group.weight_pct >= minimum_weight]
        count = int(group.manager_id.nunique())
        meaningful_count = int(meaningful.manager_id.nunique())
        if meaningful.empty:
            continue
        median_weight = float(meaningful.weight_pct.median())
        max_weight = float(group.weight_pct.max())
        if (meaningful_count >= int(cfg["core_min_managers"])
                and median_weight >= float(cfg["core_min_median_weight_pct"])):
            level = "CORE"
        elif (meaningful_count >= int(cfg["watch_min_managers"])
              and max_weight >= float(cfg["watch_min_max_weight_pct"])):
            level = "WATCH"
        else:
            continue
        positive_count = int(positive_by_cusip.get(cusip, 0))
        reducing_count = int(reducing_by_cusip.get(cusip, 0))
        if positive_count >= int(cfg["entry_min_positive_managers"]) and positive_count > reducing_count:
            direction_state = "BROAD_ACCUMULATION"
            entry_status = "DIRECTION_CONFIRMED_REVIEW"
        elif reducing_count >= int(cfg["entry_min_positive_managers"]) and reducing_count > positive_count:
            direction_state = "BROAD_DISTRIBUTION"
            entry_status = "DISTRIBUTION_REVIEW"
        elif positive_count or reducing_count:
            direction_state = "MIXED"
            entry_status = "HOLDING_CONSENSUS_ONLY"
        else:
            direction_state = "FLAT_OR_UNVERIFIED"
            entry_status = "HOLDING_CONSENSUS_ONLY"
        rows.append({
            "cusip": cusip,
            "symbol": _representative(group.symbol),
            "issuer": _representative(group.issuer),
            "candidate_level": level,
            "manager_count": count,
            "meaningful_manager_count": meaningful_count,
            "managers": "|".join(sorted(meaningful.manager_id.unique())),
            "total_disclosed_weight_pct": float(group.weight_pct.sum()),
            "median_meaningful_weight_pct": median_weight,
            "max_manager_weight_pct": max_weight,
            "positive_signal_count": positive_count,
            "reducing_manager_count": reducing_count,
            "direction_state": direction_state,
            "entry_status": entry_status,
            "evidence": (f"meaningful_weight_at_least_{minimum_weight:g}pct|"
                         f"meaningful_managers_{meaningful_count}|positive_signals_{positive_count}|"
                         f"reducing_managers_{reducing_count}"),
            "orders_sent": False,
        })
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["candidate_level", "meaningful_manager_count", "median_meaningful_weight_pct"],
        ascending=[True, False, False], ignore_index=True)
