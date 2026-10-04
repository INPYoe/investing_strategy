"""selection.py 단위 테스트."""
import pandas as pd
import pytest

from guru_tracker.selection import compute_scores, select_portfolio, TARGET_WEIGHT_PCT


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def make_counts(rows):
    """(manager_id, cusip, symbol, issuer, weight_pct, consecutive_quarters, meets_entry) 튜플 리스트 → DataFrame."""
    return pd.DataFrame(
        [{"manager_id": r[0], "cusip": r[1], "symbol": r[2], "issuer": r[3],
          "current_weight_pct": r[4], "consecutive_quarters": r[5],
          "meets_entry_condition": r[6], "M": 5,
          "current_report_date": "2026-06-30",
          "rank_in_manager": 1, "qualifies_weight": r[4] >= 5.0,
          "qualifies_top5": True}
         for r in rows]
    )


# ── compute_scores ─────────────────────────────────────────────────────────────

def test_compute_scores_empty():
    df = make_counts([])
    result = compute_scores(df)
    assert result.empty


def test_compute_scores_filters_non_eligible():
    df = make_counts([
        ("m1", "AAA", "A", "A Co", 10.0, 6, True),
        ("m1", "BBB", "B", "B Co", 8.0, 4, False),  # not eligible
    ])
    result = compute_scores(df)
    assert len(result) == 1
    assert result.iloc[0].cusip == "AAA"


def test_compute_scores_single_manager_single_company():
    # w = 10% → w_fraction = 0.10, I = min(0.10/0.20, 1) = 0.5
    # n = 8    → L = min(8/12, 1) = 0.667
    # k = 1, M = 5 → C = 0.2
    # S = 50*0.5 + 30*0.667 + 20*0.2 = 25 + 20 + 4 = 49
    df = make_counts([("m1", "AAA", "A", "A Co", 10.0, 8, True)])
    result = compute_scores(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert abs(row.I - 0.5) < 1e-3
    assert abs(row.L - 0.6667) < 1e-3
    assert abs(row.C - 0.2) < 1e-3
    assert abs(row.S - 49.0) < 0.1


def test_compute_scores_saturates_at_20pct_weight():
    # w = 25% → I = min(0.25/0.20, 1) = 1.0 (포화)
    df = make_counts([("m1", "AAA", "A", "A Co", 25.0, 12, True)])
    result = compute_scores(df)
    assert result.iloc[0].I == 1.0


def test_compute_scores_saturates_at_12_quarters():
    # n = 15 → L = min(15/12, 1) = 1.0 (포화)
    df = make_counts([("m1", "AAA", "A", "A Co", 10.0, 15, True)])
    result = compute_scores(df)
    assert result.iloc[0].L == 1.0


def test_compute_scores_two_managers_same_company():
    # 두 운용사 모두 적격 → k=2, M=5, C=0.4
    df = make_counts([
        ("m1", "AAA", "A", "A Co", 20.0, 12, True),
        ("m2", "AAA", "A", "A Co", 10.0,  8, True),
    ])
    result = compute_scores(df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.qualifying_manager_count == 2
    assert abs(row.C - 0.4) < 1e-9
    # I = mean(min(0.20/0.20,1), min(0.10/0.20,1)) = mean(1.0, 0.5) = 0.75
    assert abs(row.I - 0.75) < 1e-3
    # L = mean(min(12/12,1), min(8/12,1)) = mean(1.0, 0.667) = 0.833
    assert abs(row.L - 0.8333) < 1e-3


def test_compute_scores_sorted_by_s_desc():
    df = make_counts([
        ("m1", "LOW", "L", "Lo", 5.0,  5, True),   # lower score
        ("m1", "HIGH", "H", "Hi", 20.0, 12, True),  # higher score
    ])
    result = compute_scores(df)
    assert result.iloc[0].cusip == "HIGH"
    assert result.iloc[1].cusip == "LOW"


def test_compute_scores_tiebreak_by_manager_count():
    # Same S but different k_j (one company has 2 managers, other has 1)
    df = make_counts([
        ("m1", "A1", "A1", "A1 Co", 20.0, 12, True),
        ("m1", "B2", "B2", "B2 Co", 20.0, 12, True),
        ("m2", "B2", "B2", "B2 Co", 20.0, 12, True),
    ])
    result = compute_scores(df)
    # B2 has 2 qualifying managers → ranked higher
    assert result.iloc[0].cusip == "B2"


def test_compute_scores_tiebreak_cusip_alphabetical():
    # Same S, same k → cusip alphabetical
    df = make_counts([
        ("m1", "ZZZ", "Z", "Z Co", 20.0, 12, True),
        ("m1", "AAA", "A", "A Co", 20.0, 12, True),
    ])
    result = compute_scores(df)
    assert result.iloc[0].cusip == "AAA"


# ── select_portfolio ───────────────────────────────────────────────────────────

def _scores(cusips_scores):
    """[(cusip, S)] 리스트 → 최소 scores DataFrame."""
    rows = [{"cusip": c, "symbol": c, "issuer": c + " Co",
             "I": 1.0, "L": 1.0, "C": 1.0, "S": s,
             "qualifying_manager_count": 1, "M": 5,
             "qualifying_managers": "m1", "max_consecutive_quarters": 5}
            for c, s in cusips_scores]
    return pd.DataFrame(rows)


def test_select_portfolio_selects_up_to_eight():
    scores = _scores([(f"C{i:02d}", 100 - i) for i in range(10)])
    selected, _ = select_portfolio(scores)
    assert len(selected) == 8


def test_select_portfolio_assigns_target_weight():
    scores = _scores([("AAA", 80)])
    selected, _ = select_portfolio(scores)
    assert selected.iloc[0].target_weight_pct == TARGET_WEIGHT_PCT


def test_select_portfolio_no_caps_no_skips():
    scores = _scores([("A", 90), ("B", 80), ("C", 70)])
    selected, skipped = select_portfolio(scores)
    assert len(selected) == 3
    assert skipped.empty


def test_select_portfolio_sector_cap_blocks_third():
    # sector_cap = 30%, target = 12.5% → max 2 per sector
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech"}
    scores = _scores([("A", 90), ("B", 80), ("C", 70)])
    selected, skipped = select_portfolio(scores, sector_map=sector_map)
    # A and B selected (25% total), C blocked (37.5% > 30%)
    assert {r.cusip for r in selected.itertuples()} == {"A", "B"}
    assert len(skipped) == 1
    assert skipped.iloc[0].cusip == "C"
    assert "sector_cap:Tech" in skipped.iloc[0].skip_reason


def test_select_portfolio_group_cap_blocks_fourth():
    # group_cap = 40%, target = 12.5% → max 3 per group
    group_map = {"A": ["g1"], "B": ["g1"], "C": ["g1"], "D": ["g1"]}
    scores = _scores([("A", 90), ("B", 80), ("C", 70), ("D", 60)])
    selected, skipped = select_portfolio(scores, group_map=group_map)
    assert len(selected) == 3
    assert skipped.iloc[0].cusip == "D"
    assert "group_cap:g1" in skipped.iloc[0].skip_reason


def test_select_portfolio_unclassified_not_blocked_by_sector_cap():
    sector_map = {"A": "Tech", "B": "Tech"}
    scores = _scores([("A", 90), ("B", 80), ("C", 70)])
    # C has no sector entry → UNCLASSIFIED → not subject to sector cap
    selected, skipped = select_portfolio(scores, sector_map=sector_map)
    assert "C" in {r.cusip for r in selected.itertuples()}
    assert skipped.empty


def test_select_portfolio_multi_group_both_caps_apply():
    # Company D belongs to g1 AND g2. g1 is already at 37.5% (3 companies), g2 at 25% (2 companies).
    group_map = {
        "A": ["g1"], "B": ["g1"], "C": ["g1"],  # 3 in g1 = 37.5%
        "X": ["g2"], "Y": ["g2"],                # 2 in g2 = 25%
        "D": ["g1", "g2"],                       # would push g1 to 50% → blocked
    }
    scores = _scores([("A", 90), ("B", 80), ("C", 70), ("X", 65), ("Y", 60), ("D", 55)])
    selected, skipped = select_portfolio(scores, group_map=group_map)
    selected_cusips = {r.cusip for r in selected.itertuples()}
    assert "D" not in selected_cusips
    d_skip = skipped.loc[skipped.cusip.eq("D")]
    assert not d_skip.empty
    assert "group_cap:g1" in d_skip.iloc[0].skip_reason


def test_select_portfolio_skipped_candidate_filled_by_next():
    # A and B are both Tech, C is Finance — after A and B exhaust sector cap,
    # C should still be selected
    sector_map = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Finance"}
    scores = _scores([("A", 90), ("B", 80), ("C", 70), ("D", 60)])
    selected, skipped = select_portfolio(scores, sector_map=sector_map)
    selected_cusips = {r.cusip for r in selected.itertuples()}
    assert "D" in selected_cusips
    assert "C" not in selected_cusips


def test_select_portfolio_empty_scores():
    selected, skipped = select_portfolio(pd.DataFrame())
    assert selected.empty
    assert skipped.empty
