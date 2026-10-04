import pandas as pd

from guru_tracker.consensus import build_consensus_candidates


POLICY = {"consensus": {"meaningful_manager_weight_pct": 1, "core_min_managers": 3,
                         "core_min_median_weight_pct": 2, "watch_min_managers": 2,
                         "watch_min_max_weight_pct": 3, "entry_min_positive_managers": 2}}


def holdings(rows):
    return pd.DataFrame(rows, columns=["cusip", "symbol", "issuer", "manager_id", "weight_pct"])


def signals(rows=()):
    return pd.DataFrame(rows, columns=["cusip", "manager_id", "signal"])


def test_core_requires_three_meaningful_managers_and_median_weight():
    frame = holdings([("1", "AAA", "A", "m1", 8), ("1", "AAA", "A", "m2", 3),
                      ("1", "AAA", "A", "m3", 2), ("1", "AAA", "A", "m4", .2)])
    row = build_consensus_candidates(frame, signals(), POLICY).iloc[0]
    assert row.candidate_level == "CORE"
    assert row.manager_count == 4
    assert row.meaningful_manager_count == 3
    assert row.entry_status == "HOLDING_CONSENSUS_ONLY"


def test_watch_and_direction_confirmation_are_separate_from_order():
    frame = holdings([("2", "BBB", "B", "m1", 7), ("2", "BBB", "B", "m2", 1.5)])
    change = signals([("2", "m1", "NEW_ENTRY_CANDIDATE"),
                      ("2", "m2", "INCREASE_ENTRY_CANDIDATE")])
    row = build_consensus_candidates(frame, change, POLICY).iloc[0]
    assert row.candidate_level == "WATCH"
    assert row.positive_signal_count == 2
    assert row.entry_status == "DIRECTION_CONFIRMED_REVIEW"
    assert row.orders_sent == False


def test_tiny_positions_do_not_create_consensus():
    frame = holdings([("3", "CCC", "C", "m1", 5), ("3", "CCC", "C", "m2", .4),
                      ("3", "CCC", "C", "m3", .2)])
    assert build_consensus_candidates(frame, signals(), POLICY).empty


def test_two_accumulating_managers_confirm_consensus_direction():
    frame = holdings([("4", "DDD", "D", "m1", 5), ("4", "DDD", "D", "m2", 3),
                      ("4", "DDD", "D", "m3", 2)])
    changes = pd.DataFrame([
        ("4", "m1", "NO_ACTION", "ACCUMULATING"),
        ("4", "m2", "NO_ACTION", "ACCUMULATING"),
        ("4", "m3", "NO_ACTION", "REDUCING"),
    ], columns=["cusip", "manager_id", "signal", "holding_direction"])
    row = build_consensus_candidates(frame, changes, POLICY).iloc[0]
    assert row.direction_state == "BROAD_ACCUMULATION"
    assert row.entry_status == "DIRECTION_CONFIRMED_REVIEW"
