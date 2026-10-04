import pandas as pd
import pytest

from guru_tracker.multi_guru import (
    compare_manager,
    consecutive_count,
    expected_report_date,
    _is_adjacent_quarter,
    build_consecutive_counts,
)


def frame(rows):
    return pd.DataFrame(rows, columns=["cusip", "symbol", "issuer", "weight_pct", "shares"])


MANAGER = {"id": "m", "name": "Manager", "lead_name": "Team"}
FILING_OLD = {"reportDate": "2026-03-31", "acceptanceDateTime": "2026-05-15T20:00:00Z", "accessionNumber": "old"}
FILING_NEW = {"reportDate": "2026-06-30", "acceptanceDateTime": "2026-08-14T20:00:00Z", "accessionNumber": "new"}
POLICY = {"entry": {"new_position_min_weight_pct": 3}}


def test_only_new_three_percent_is_tomorrow_actionable():
    old = frame([("A", "AAA", "A Co", 2, 100), ("X", "XXX", "X Co", 6, 100)])
    new = frame([("A", "AAA", "A Co", 4, 200), ("B", "BBB", "B Co", 3, 50)])
    rows = compare_manager(MANAGER, FILING_OLD, FILING_NEW, old, new, POLICY)
    by_symbol = {r["symbol"]: r for r in rows}
    assert by_symbol["BBB"]["signal"] == "NEW_ENTRY_CANDIDATE"
    assert by_symbol["BBB"]["actionable_for_tomorrow_plan"] is True
    assert by_symbol["AAA"]["signal"] == "REVIEW"
    assert by_symbol["AAA"]["actionable_for_tomorrow_plan"] is False
    assert by_symbol["XXX"]["signal"] == "REVIEW"


def test_ordinary_held_change_is_not_actionable():
    old = frame([("A", "AAA", "A Co", 4, 100)])
    new = frame([("A", "AAA", "A Co", 5, 110)])
    row = compare_manager(MANAGER, FILING_OLD, FILING_NEW, old, new, POLICY)[0]
    assert row["signal"] == "REVIEW"
    assert row["actionable_for_tomorrow_plan"] is False
    assert row["holding_direction"] == "CORPORATE_ACTION_REVIEW"


def test_large_ratio_change_requires_corporate_action_review_for_direction():
    old = frame([("A", "AAA", "A Co", 1, 100)])
    new = frame([("A", "AAA", "A Co", 3, 250)])
    row = compare_manager(MANAGER, FILING_OLD, FILING_NEW, old, new, POLICY)[0]
    assert row["holding_direction"] == "CORPORATE_ACTION_REVIEW"
    assert row["direction_quality"] == "CORPORATE_ACTIONS_UNVERIFIED"


def test_split_factor_one_resolves_direction():
    old = frame([("A", "AAA", "A Co", 4, 100)])
    new = frame([("A", "AAA", "A Co", 5, 110)])
    row = compare_manager(MANAGER, FILING_OLD, FILING_NEW, old, new, POLICY, split_factors={"A": 1.0})[0]
    assert row["holding_direction"] == "ACCUMULATING"
    assert row["direction_quality"] == "YAHOO_SPLITS_VERIFIED"
    assert row["split_factor"] == 1.0


def test_split_factor_two_doubles_shares_flat_weight():
    # 2:1 split → shares doubled, weight unchanged → MIXED_OR_FLAT
    old = frame([("A", "AAA", "A Co", 5, 100)])
    new = frame([("A", "AAA", "A Co", 5, 200)])
    row = compare_manager(MANAGER, FILING_OLD, FILING_NEW, old, new, POLICY, split_factors={"A": 2.0})[0]
    assert row["holding_direction"] == "MIXED_OR_FLAT"
    assert row["direction_quality"] == "YAHOO_SPLITS_VERIFIED"


def test_expected_latest_period_observes_45_day_deadline():
    assert expected_report_date("2026-09-21") == "2026-06-30"
    assert expected_report_date("2026-08-13") == "2026-03-31"
    assert expected_report_date("2026-08-14") == "2026-06-30"


# --- _is_adjacent_quarter ---

def test_adjacent_quarter_q1_q2():
    assert _is_adjacent_quarter("2026-03-31", "2026-06-30") is True


def test_adjacent_quarter_q4_q1():
    assert _is_adjacent_quarter("2025-12-31", "2026-03-31") is True


def test_non_adjacent_quarters():
    assert _is_adjacent_quarter("2025-12-31", "2026-06-30") is False  # 2 quarters apart


# --- consecutive_count ---

def _make_pairs(periods_with_cusips):
    """Build minimal pairs list: [(filing_dict, holdings_df), ...] from [(period, [cusip, ...]), ...]"""
    result = []
    for period, cusips in periods_with_cusips:
        filing = {"reportDate": period, "accessionNumber": period.replace("-", ""), "acceptanceDateTime": ""}
        df = pd.DataFrame(
            [(c, c, "Co", 5.0, 100) for c in cusips],
            columns=["cusip", "symbol", "issuer", "weight_pct", "shares"],
        )
        result.append((filing, df))
    return result


def test_consecutive_count_all_five():
    pairs = _make_pairs([
        ("2025-06-30", ["A", "B"]),
        ("2025-09-30", ["A", "B"]),
        ("2025-12-31", ["A", "B"]),
        ("2026-03-31", ["A", "B"]),
        ("2026-06-30", ["A", "B"]),
    ])
    assert consecutive_count(pairs, "A") == 5
    assert consecutive_count(pairs, "B") == 5


def test_consecutive_count_missing_in_middle():
    # A present Q1–Q2 only; B present Q3–Q5 only; "gap" breaks the count from newest
    pairs = _make_pairs([
        ("2025-06-30", ["A"]),
        ("2025-09-30", ["A"]),
        ("2025-12-31", ["B"]),
        ("2026-03-31", ["B"]),
        ("2026-06-30", ["B"]),
    ])
    assert consecutive_count(pairs, "B") == 3  # Q4, Q5, Q6 consecutive from newest
    assert consecutive_count(pairs, "A") == 0  # not in latest period


def test_consecutive_count_period_gap_breaks_count():
    # Skip Q3 2025, creating a non-adjacent gap between Q2 and Q4
    pairs = _make_pairs([
        ("2025-06-30", ["A"]),   # Q2
        ("2025-12-31", ["A"]),   # Q4 — gap: Q3 missing
        ("2026-03-31", ["A"]),
        ("2026-06-30", ["A"]),
    ])
    # From newest: Q2 2026 → Q1 2026 adjacent ✓ → Q4 2025 adjacent ✓ → Q2 2025 not adjacent (gap) → stop
    assert consecutive_count(pairs, "A") == 3


def test_consecutive_count_absent_in_latest():
    pairs = _make_pairs([
        ("2025-06-30", ["A"]),
        ("2025-09-30", ["A"]),
        ("2025-12-31", ["A"]),
        ("2026-03-31", ["A"]),
        ("2026-06-30", []),       # A absent in latest
    ])
    assert consecutive_count(pairs, "A") == 0


# --- build_consecutive_counts ---

def test_build_consecutive_counts_meets_entry_condition():
    cohort = [{"id": "m1"}, {"id": "m2"}]
    pairs_m1 = _make_pairs([
        ("2025-06-30", ["X"]),
        ("2025-09-30", ["X"]),
        ("2025-12-31", ["X"]),
        ("2026-03-31", ["X"]),
        ("2026-06-30", ["X"]),   # weight = 5%, so qualifies_weight True
    ])
    pairs_m2 = _make_pairs([
        ("2026-03-31", ["X"]),
        ("2026-06-30", ["X"]),   # only 2 quarters
    ])
    history = {"m1": pairs_m1, "m2": pairs_m2}
    df = build_consecutive_counts(history, cohort)
    m1_row = df.loc[df.manager_id.eq("m1") & df.cusip.eq("X")].iloc[0]
    assert m1_row.consecutive_quarters == 5
    assert bool(m1_row.meets_entry_condition) is True
    m2_row = df.loc[df.manager_id.eq("m2") & df.cusip.eq("X")].iloc[0]
    assert m2_row.consecutive_quarters == 2
    assert bool(m2_row.meets_entry_condition) is False


def test_build_consecutive_counts_top5_qualifies():
    cohort = [{"id": "m1"}]
    # 6 holdings, X is rank 5 (just inside top5), weight = 4% (below 5% threshold)
    holdings = [("A", "A", "A", 20.0, 100), ("B", "B", "B", 15.0, 100),
                ("C", "C", "C", 12.0, 100), ("D", "D", "D", 10.0, 100),
                ("X", "X", "X",  4.0, 100), ("Z", "Z", "Z",  3.0, 100)]
    pairs = []
    for period in ["2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30"]:
        filing = {"reportDate": period, "accessionNumber": period.replace("-", ""), "acceptanceDateTime": ""}
        df = pd.DataFrame(holdings, columns=["cusip", "symbol", "issuer", "weight_pct", "shares"])
        pairs.append((filing, df))
    history = {"m1": pairs}
    result = build_consecutive_counts(history, cohort)
    x_row = result.loc[result.cusip.eq("X")].iloc[0]
    assert x_row.rank_in_manager == 5
    assert bool(x_row.qualifies_top5) is True
    assert bool(x_row.qualifies_weight) is False
    assert bool(x_row.meets_entry_condition) is True  # top5 AND 5 consecutive quarters

    z_row = result.loc[result.cusip.eq("Z")].iloc[0]
    assert z_row.rank_in_manager == 6
    assert bool(z_row.qualifies_top5) is False
    assert bool(z_row.meets_entry_condition) is False
