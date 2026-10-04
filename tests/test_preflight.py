"""preflight.py 단위 테스트."""
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from guru_tracker.preflight import (
    build_order_for_action,
    night_preflight,
    resolve_stage,
    session_open_kst,
)

KST = ZoneInfo("Asia/Seoul")


def _dt_kst(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=KST)


def _mock_broker(
    quotes=None,
    holdings=None,
    pending_order_count=0,
    orderable_usd="10000.00",
    account_fingerprint="TEST_FP",
    observed_at=None,
):
    if quotes is None:
        quotes = {}
    if holdings is None:
        holdings = []
    if observed_at is None:
        observed_at = datetime.now(timezone.utc).isoformat()

    def snapshot(instruments):
        return {
            "observed_at": observed_at,
            "account_fingerprint": account_fingerprint,
            "holdings": holdings,
            "pending_order_count": pending_order_count,
            "available_usd": "10000.00",
            "quotes": quotes,
        }

    def integrated_margin_snapshot():
        return {
            "subscription": "통합증거금신청",
            "usd_integrated_orderable": orderable_usd,
            "usd_reference_fx": "1350.00",
        }

    return SimpleNamespace(snapshot=snapshot, integrated_margin_snapshot=integrated_margin_snapshot,
                           fingerprint=account_fingerprint)


def _verified(symbols):
    return [{"symbol": s, "cusip": (s + "000000000")[:9], "exchange": "NASD",
             "standard_code": "US" + (s + "000000000")[:9]} for s in symbols]


def _plan(actions, verified_symbols=None, asof="2026-06-30"):
    if verified_symbols is None:
        verified_symbols = list({a["symbol"] for a in actions})
    return {
        "asof": asof,
        "status": "READY_FOR_USER_ORDER_ENTRY_AFTER_LIVE_QUOTE_CHECK",
        "verified_instruments": _verified(verified_symbols),
        "rebalance_actions": actions,
    }


def _action(symbol, action_type, delta_qty, ref_price="100.00", current_qty=0, target_qty=None, **kwargs):
    if target_qty is None:
        target_qty = delta_qty
    return {"action": action_type, "symbol": symbol, "delta_qty": delta_qty,
            "reference_price_usd": ref_price, "current_weight_pct": 0.0,
            "target_weight_pct": 12.5, "current_qty": current_qty,
            "target_qty": target_qty, "cusip": (symbol + "000000000")[:9], **kwargs}


# ── session_open_kst ──────────────────────────────────────────────────────────

def test_session_open_during_regular_hours():
    # September EDT (UTC-4): 23:00 KST → UTC 14:00 → ET 10:00 AM → open
    assert session_open_kst(_dt_kst(2026, 9, 22, 23, 0)) is True


def test_session_closed_before_open():
    # 21:00 KST → UTC 12:00 → ET 8:00 AM → before 9:30 AM
    assert session_open_kst(_dt_kst(2026, 9, 22, 21, 0)) is False


def test_session_closed_after_market_close():
    # 2026-09-23 06:00 KST → UTC 2026-09-22 21:00 → ET 17:00 → after 16:00
    assert session_open_kst(_dt_kst(2026, 9, 23, 6, 0)) is False


def test_session_open_at_open_boundary():
    # 22:30 KST (EDT) → UTC 13:30 → ET 9:30 AM → exactly at open → True
    assert session_open_kst(_dt_kst(2026, 9, 22, 22, 30)) is True


def test_session_closed_at_close_boundary():
    # 05:00 KST+1 (EDT) → UTC 20:00 → ET 16:00 → exactly at close → False
    assert session_open_kst(_dt_kst(2026, 9, 23, 5, 0)) is False


# ── resolve_stage ─────────────────────────────────────────────────────────────

def test_auto_stage_morning():
    assert resolve_stage("auto", _dt_kst(2026, 9, 22, 9, 5)) == "morning"


def test_auto_stage_night():
    assert resolve_stage("auto", _dt_kst(2026, 9, 22, 22, 0)) == "night"


def test_explicit_stage_overrides_time():
    assert resolve_stage("morning", _dt_kst(2026, 9, 22, 22, 0)) == "morning"
    assert resolve_stage("night", _dt_kst(2026, 9, 22, 9, 0)) == "night"


# ── build_order_for_action ────────────────────────────────────────────────────

def test_build_order_new_entry_is_buy():
    order = build_order_for_action(_action("AAA", "NEW_ENTRY", 10), "NASD", "100.00", "2026-06-30")
    assert order["side"] == "BUY"
    assert order["quantity"] == 10
    assert order["limit_price_usd"] == "100.00"
    assert order["intent_id"] == "2026-06-30:NEW_ENTRY:AAA:10"
    assert order["exchange"] == "NASD"


def test_build_order_sell_rebalance_is_sell():
    order = build_order_for_action(_action("BBB", "SELL_REBALANCE", 3), "NYSE", "200.00", "2026-06-30")
    assert order["side"] == "SELL"
    assert order["exchange"] == "NYSE"


def test_build_order_buy_rebalance_is_buy():
    order = build_order_for_action(_action("CCC", "BUY_REBALANCE", 4), "NASD", "50.00", "2026-06-30")
    assert order["side"] == "BUY"


# ── night_preflight ───────────────────────────────────────────────────────────

def test_hold_rows_excluded():
    actions = [
        _action("AAA", "NEW_ENTRY", 10),
        _action("BBB", "HOLD", 0, current_qty=10, target_qty=10),
    ]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    plan = _plan(actions, verified_symbols=["AAA", "BBB"])
    results = night_preflight(plan, broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    symbols = [r["symbol"] for r in results]
    assert "BBB" not in symbols
    assert "AAA" in symbols


def test_exit_sell_excluded():
    actions = [_action("OLD", "EXIT_SELL", 5)]
    broker = _mock_broker(
        quotes={"OLD": {"price": "100.00", "exchange": "NASD"}},
    )
    plan = _plan(actions, verified_symbols=[])
    results = night_preflight(plan, broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results == []


def test_all_preflight_fields_true_when_conditions_met():
    actions = [_action("AAA", "NEW_ENTRY", 5)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    plan = _plan(actions)
    results = night_preflight(plan, broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert len(results) == 1
    pre = results[0]["preflight"]
    assert pre["instrument_verified"] is True
    assert pre["session_open_verified"] is True
    assert pre["quote_verified"] is True
    assert pre["cash_and_position_verified"] is True
    assert pre["no_pending_orders"] is True
    assert pre["plan_reviewed"] is True


def test_quote_verified_false_when_price_above_103pct():
    # Live 105 > 100 * 1.03 = 103
    actions = [_action("AAA", "NEW_ENTRY", 5, ref_price="100.00")]
    broker = _mock_broker(
        quotes={"AAA": {"price": "105.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["quote_verified"] is False


def test_quote_verified_true_at_103pct_boundary():
    actions = [_action("AAA", "NEW_ENTRY", 5, ref_price="100.00")]
    broker = _mock_broker(
        quotes={"AAA": {"price": "103.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["quote_verified"] is True


def test_cash_and_position_verified_false_insufficient_orderable():
    # Buy 10 @ $100 = $1000 notional, but only $500 orderable
    actions = [_action("AAA", "NEW_ENTRY", 10, ref_price="100.00")]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="500.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["cash_and_position_verified"] is False


def test_cash_and_position_verified_false_for_sell_insufficient_qty():
    # Need to sell 10, but only holding 5
    actions = [_action("AAA", "SELL_REBALANCE", 10, current_qty=15, target_qty=12)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        holdings=[{"symbol": "AAA", "quantity": "5", "market_value_usd": "500"}],
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["cash_and_position_verified"] is False


def test_cash_and_position_verified_true_for_sell_with_sufficient_qty():
    actions = [_action("AAA", "SELL_REBALANCE", 3, current_qty=15, target_qty=12)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        holdings=[{"symbol": "AAA", "quantity": "15", "market_value_usd": "1500"}],
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["cash_and_position_verified"] is True


def test_no_pending_orders_false_when_pending():
    actions = [_action("AAA", "NEW_ENTRY", 5)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        pending_order_count=2,
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["no_pending_orders"] is False


def test_plan_reviewed_false_when_not_reviewed():
    actions = [_action("AAA", "NEW_ENTRY", 5)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=False, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["plan_reviewed"] is False


def test_order_sha256_matches_order():
    actions = [_action("AAA", "BUY_REBALANCE", 3, ref_price="100.00", current_qty=8, target_qty=12)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    entry = results[0]
    expected_sha = hashlib.sha256(json.dumps(entry["order"], sort_keys=True).encode()).hexdigest()
    assert entry["preflight"]["order_sha256"] == expected_sha


def test_instrument_verified_false_when_not_in_plan():
    actions = [_action("GHOST", "NEW_ENTRY", 5)]
    broker = _mock_broker(
        quotes={"GHOST": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    # verified_instruments is empty
    plan = _plan(actions, verified_symbols=[])
    results = night_preflight(plan, broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results[0]["preflight"]["instrument_verified"] is False


def test_session_open_false_before_market():
    # 20:00 KST → ET 7:00 AM → closed
    actions = [_action("AAA", "NEW_ENTRY", 5)]
    broker = _mock_broker(
        quotes={"AAA": {"price": "100.00", "exchange": "NASD"}},
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 20, 0))
    assert results[0]["preflight"]["session_open_verified"] is False


def test_empty_plan_returns_empty():
    broker = _mock_broker()
    plan = _plan([], verified_symbols=[])
    results = night_preflight(plan, broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert results == []


def test_multiple_orders_all_generated():
    actions = [
        _action("AAA", "NEW_ENTRY", 10),
        _action("BBB", "BUY_REBALANCE", 3, current_qty=8, target_qty=12),
        _action("CCC", "SELL_REBALANCE", 2, current_qty=15, target_qty=12),
    ]
    broker = _mock_broker(
        quotes={
            "AAA": {"price": "100.00", "exchange": "NASD"},
            "BBB": {"price": "50.00", "exchange": "NASD"},
            "CCC": {"price": "200.00", "exchange": "NASD"},
        },
        holdings=[
            {"symbol": "BBB", "quantity": "8", "market_value_usd": "400"},
            {"symbol": "CCC", "quantity": "15", "market_value_usd": "3000"},
        ],
        orderable_usd="2000.00",
    )
    results = night_preflight(_plan(actions), broker, reviewed=True, now=_dt_kst(2026, 9, 22, 23, 0))
    assert len(results) == 3
    action_map = {r["symbol"]: r["order"]["side"] for r in results}
    assert action_map["AAA"] == "BUY"
    assert action_map["BBB"] == "BUY"
    assert action_map["CCC"] == "SELL"
