import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from guru_tracker.preflight import resolve_stage
from guru_tracker.tomorrow_plan import (
    build_rebalance_plan,
    cached_instrument,
    full_investment_quantities,
    quantity_for,
)


# ── quantity_for ───────────────────────────────────────────────────────────────

def test_quantity_rounds_down_without_leverage():
    assert quantity_for(1000, 20, 51) == 3
    assert quantity_for(1000, 20, 500) == 0


# ── full_investment_quantities ─────────────────────────────────────────────────

def test_full_investment_allocation_uses_residual_without_borrowing():
    targets = {"AAA": 40, "BBB": 31, "CCC": 29}
    quotes = {"AAA": {"price": "154.48"}, "BBB": {"price": "115.05"},
              "CCC": {"price": "201.92"}}
    quantities = full_investment_quantities("7766.43", targets, quotes)
    invested = sum(float(quotes[s]["price"]) * quantities[s] for s in quantities)
    assert invested <= 7766.43
    assert invested / 7766.43 >= .98
    assert 7766.43 - invested < min(float(q["price"]) for q in quotes.values())


# ── cached_instrument ──────────────────────────────────────────────────────────

def test_recent_exact_cusip_instrument_cache_only(tmp_path):
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({
        "created_at": (now - timedelta(hours=2)).isoformat(),
        "verified_instruments": [{"symbol": "ABC", "cusip": "123456789",
                                  "standard_code": "US1234567890", "exchange": "NASD"}],
    }))
    item = cached_instrument(path, "ABC", "123456789", now)
    assert item["verification_source"] == "KIS_EXACT_CUSIP_CACHE_LT_24H"
    with pytest.raises(RuntimeError, match="NOT_UNIQUE"):
        cached_instrument(path, "ABC", "987654321", now)


def test_expired_instrument_cache_is_rejected(tmp_path):
    now = datetime(2026, 9, 21, 12, tzinfo=timezone.utc)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"created_at": (now - timedelta(hours=25)).isoformat(),
                                "verified_instruments": []}))
    with pytest.raises(RuntimeError, match="EXPIRED"):
        cached_instrument(path, "ABC", "123456789", now)


# ── preflight stage ────────────────────────────────────────────────────────────

def test_auto_stage_uses_seoul_session_boundary():
    tz = ZoneInfo("Asia/Seoul")
    assert resolve_stage("auto", datetime(2026, 9, 22, 9, 5, tzinfo=tz)) == "morning"
    assert resolve_stage("auto", datetime(2026, 9, 22, 22, 5, tzinfo=tz)) == "night"
    assert resolve_stage("morning", datetime(2026, 9, 22, 22, 5, tzinfo=tz)) == "morning"


# ── build_rebalance_plan helpers ───────────────────────────────────────────────

def _target(symbol, cusip):
    return {"symbol": symbol, "cusip": cusip, "target_weight_pct": 12.5}


def _holding(symbol, qty, value_usd):
    return {"symbol": symbol, "quantity": str(qty), "market_value_usd": str(value_usd)}


def _quote(symbol, price):
    return {symbol: {"price": str(price), "exchange": "NASD"}}


def _equity(usd):
    return Decimal(str(usd))


# ── build_rebalance_plan ───────────────────────────────────────────────────────

def test_new_entry_when_not_currently_held():
    # $10,000 equity, 12.5% target → $1,250, price $100 → 12 shares
    result = build_rebalance_plan(
        [_target("AAA", "111111111")],
        [],
        _equity(10000),
        _quote("AAA", 100),
    )
    assert len(result) == 1
    row = result[0]
    assert row["action"] == "NEW_ENTRY"
    assert row["current_qty"] == 0
    assert row["target_qty"] == 12
    assert row["delta_qty"] == 12
    assert row["current_weight_pct"] == 0.0


def test_hold_when_within_band():
    # $10,000 equity, holding 12 shares at $100 = $1,200 = 12.0% (within 9.375%~15.625%)
    result = build_rebalance_plan(
        [_target("AAA", "111111111")],
        [_holding("AAA", 12, 1200)],
        _equity(10000),
        _quote("AAA", 100),
    )
    assert result[0]["action"] == "HOLD"
    assert result[0]["delta_qty"] == 0


def test_buy_rebalance_below_lower_band():
    # $10,000 equity, holding 8 shares at $100 = $800 = 8.0% (below 9.375%)
    result = build_rebalance_plan(
        [_target("AAA", "111111111")],
        [_holding("AAA", 8, 800)],
        _equity(10000),
        _quote("AAA", 100),
    )
    row = result[0]
    assert row["action"] == "BUY_REBALANCE"
    assert row["current_weight_pct"] == 8.0
    # target_qty = floor(10000 * 12.5% / 100) = 12, delta = 12 - 8 = 4
    assert row["target_qty"] == 12
    assert row["delta_qty"] == 4


def test_sell_rebalance_above_upper_band():
    # $10,000 equity, holding 17 shares at $100 = $1,700 = 17.0% (above 15.625%)
    result = build_rebalance_plan(
        [_target("AAA", "111111111")],
        [_holding("AAA", 17, 1700)],
        _equity(10000),
        _quote("AAA", 100),
    )
    row = result[0]
    assert row["action"] == "SELL_REBALANCE"
    assert row["current_weight_pct"] == 17.0
    # target_qty = 12, delta = 17 - 12 = 5
    assert row["target_qty"] == 12
    assert row["delta_qty"] == 5


def test_exit_sell_not_in_target():
    # HOLD AAA (in target), EXIT BBB (not in target)
    result = build_rebalance_plan(
        [_target("AAA", "111111111")],
        [_holding("AAA", 12, 1200), _holding("BBB", 10, 900)],
        _equity(10000),
        {**_quote("AAA", 100), **_quote("BBB", 90)},
    )
    by_symbol = {r["symbol"]: r for r in result}
    assert by_symbol["AAA"]["action"] == "HOLD"
    assert by_symbol["BBB"]["action"] == "EXIT_SELL"
    assert by_symbol["BBB"]["delta_qty"] == 10
    assert by_symbol["BBB"]["target_weight_pct"] == 0.0


def test_exit_sell_uses_known_price_if_available():
    result = build_rebalance_plan(
        [], [_holding("OLD", 5, 500)], _equity(10000), _quote("OLD", 100)
    )
    assert result[0]["reference_price_usd"] == "100"


def test_exit_sell_marks_unknown_price_if_not_in_quotes():
    result = build_rebalance_plan(
        [], [_holding("OLD", 5, 500)], _equity(10000), {}
    )
    assert result[0]["reference_price_usd"] == "PRICE_NOT_PREFETCHED"


def test_lower_band_boundary_is_hold():
    # exactly at lower band: 9.375% = 9.375% of $10,000 = $937.5
    # holding 9 shares at $100 = $900 = 9.0% → below band → BUY_REBALANCE
    # holding 10 shares at $100 = $1,000 = 10.0% → within band → HOLD
    result_below = build_rebalance_plan(
        [_target("A", "111111111")], [_holding("A", 9, 900)], _equity(10000), _quote("A", 100)
    )
    result_inside = build_rebalance_plan(
        [_target("A", "111111111")], [_holding("A", 10, 1000)], _equity(10000), _quote("A", 100)
    )
    assert result_below[0]["action"] == "BUY_REBALANCE"
    assert result_inside[0]["action"] == "HOLD"


def test_upper_band_boundary_is_hold():
    # 15.625% of $10,000 = $1,562.5
    # holding 15 shares at $100 = $1,500 = 15.0% → within band → HOLD
    # holding 16 shares at $100 = $1,600 = 16.0% → above band → SELL_REBALANCE
    result_inside = build_rebalance_plan(
        [_target("A", "111111111")], [_holding("A", 15, 1500)], _equity(10000), _quote("A", 100)
    )
    result_above = build_rebalance_plan(
        [_target("A", "111111111")], [_holding("A", 16, 1600)], _equity(10000), _quote("A", 100)
    )
    assert result_inside[0]["action"] == "HOLD"
    assert result_above[0]["action"] == "SELL_REBALANCE"


def test_missing_quote_raises():
    with pytest.raises(ValueError, match="QUOTE_MISSING"):
        build_rebalance_plan([_target("AAA", "111111111")], [], _equity(10000), {})


def test_empty_target_empty_holdings_returns_empty():
    result = build_rebalance_plan([], [], _equity(10000), {})
    assert result == []


def test_multiple_targets_all_new():
    targets = [_target("A", "AAA"), _target("B", "BBB"), _target("C", "CCC")]
    quotes = {s: {"price": "100", "exchange": "NASD"} for s in "ABC"}
    result = build_rebalance_plan(targets, [], _equity(10000), quotes)
    assert all(r["action"] == "NEW_ENTRY" for r in result)
    assert len(result) == 3
