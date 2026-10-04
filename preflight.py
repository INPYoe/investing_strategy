"""미국 야간 정규장 전·중 상태를 읽기 전용으로 점검하고 OrderGateway 프리플라이트 딕셔너리를 생성한다."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from guru_tracker.kis_readonly import KISReadOnly, number
from guru_tracker.study import ROOT


def resolve_stage(stage: str, now: datetime) -> str:
    """예약 실행은 서울 시각 오후 3시를 경계로 오전·야간 점검을 나눈다."""
    if stage != "auto":
        return stage
    return "morning" if now.hour < 15 else "night"


def session_open_kst(now_kst: datetime) -> bool:
    """미국 정규장(ET 09:30~16:00)이 열려 있는지 확인한다. DST는 ZoneInfo가 처리한다."""
    now_et = now_kst.astimezone(ZoneInfo("America/New_York"))
    t = now_et.time()
    return time(9, 30) <= t < time(16, 0)


def _intent_id(asof: str, action: str, symbol: str, delta_qty: int) -> str:
    return f"{asof}:{action}:{symbol}:{delta_qty}"


def build_order_for_action(row: dict, exchange: str, limit_price_usd: str, asof: str) -> dict:
    """rebalance_actions 행 하나를 OrderGateway 주문 딕셔너리로 변환한다."""
    side = "BUY" if row["action"] in ("NEW_ENTRY", "BUY_REBALANCE") else "SELL"
    return {
        "intent_id": _intent_id(asof, row["action"], row["symbol"], row["delta_qty"]),
        "symbol": row["symbol"],
        "exchange": exchange,
        "side": side,
        "quantity": row["delta_qty"],
        "limit_price_usd": limit_price_usd,
    }


def night_preflight(
    plan: dict, broker: KISReadOnly, *, reviewed: bool, now: datetime | None = None
) -> list[dict]:
    """plan.json의 액션 가능 행마다 OrderGateway.submit()에 넘길 딕셔너리를 생성한다.

    반환: [{"order": {...}, "preflight": {...}, "action": str, "symbol": str}, ...]
    HOLD·EXIT_SELL 행은 제외한다.
    """
    now_kst = (now or datetime.now(ZoneInfo("Asia/Seoul"))).astimezone(ZoneInfo("Asia/Seoul"))

    instruments = {
        item["cusip"]: {"symbol": item["symbol"], "exchange": item["exchange"]}
        for item in plan.get("verified_instruments", [])
    }

    snapshot = broker.snapshot(instruments)
    margin = broker.integrated_margin_snapshot()

    session_open = session_open_kst(now_kst)
    no_pending = snapshot["pending_order_count"] == 0
    orderable_usd = number(margin["usd_integrated_orderable"])

    verified_by_symbol = {item["symbol"]: item for item in plan.get("verified_instruments", [])}
    holdings_by_symbol = {h["symbol"]: int(number(h["quantity"])) for h in snapshot["holdings"]}

    actionable = [
        r for r in plan.get("rebalance_actions", [])
        if r["action"] in ("NEW_ENTRY", "BUY_REBALANCE", "SELL_REBALANCE")
    ]

    total_buy_notional = Decimal("0")
    for r in actionable:
        if r["action"] in ("NEW_ENTRY", "BUY_REBALANCE") and r["symbol"] in snapshot["quotes"]:
            total_buy_notional += number(snapshot["quotes"][r["symbol"]]["price"]) * r["delta_qty"]

    asof = plan["asof"]
    results = []

    for row in actionable:
        symbol = row["symbol"]
        action = row["action"]
        is_buy = action in ("NEW_ENTRY", "BUY_REBALANCE")

        instrument_verified = symbol in verified_by_symbol

        if symbol in snapshot["quotes"]:
            live_price_dec = number(snapshot["quotes"][symbol]["price"]).quantize(Decimal("0.01"))
            live_price = str(live_price_dec)
            ref = row["reference_price_usd"]
            quote_verified = bool(
                ref != "PRICE_NOT_PREFETCHED"
                and live_price_dec <= number(ref) * Decimal("1.03")
            )
        else:
            live_price = row["reference_price_usd"]
            quote_verified = False

        if is_buy:
            cash_and_position_verified = bool(orderable_usd >= total_buy_notional)
        else:
            held_qty = holdings_by_symbol.get(symbol, 0)
            cash_and_position_verified = held_qty >= row["delta_qty"]

        exchange = verified_by_symbol[symbol]["exchange"] if symbol in verified_by_symbol else "NASD"
        order = build_order_for_action(row, exchange, live_price, asof)
        sha = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()

        results.append({
            "order": order,
            "preflight": {
                "observed_at": snapshot["observed_at"],
                "account_fingerprint": snapshot["account_fingerprint"],
                "instrument_verified": instrument_verified,
                "session_open_verified": session_open,
                "quote_verified": quote_verified,
                "cash_and_position_verified": cash_and_position_verified,
                "no_pending_orders": no_pending,
                "plan_reviewed": reviewed,
                "order_sha256": sha,
            },
            "action": action,
            "symbol": symbol,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="구루 투자 전환 읽기 전용 프리플라이트")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--stage", choices=["morning", "night", "auto"], default="auto")
    parser.add_argument("--reviewed", action="store_true",
                        help="사용자가 plan.json을 직접 검토한 경우 지정한다.")
    args = parser.parse_args()

    now = datetime.now(ZoneInfo("Asia/Seoul"))
    if now.date().isoformat() != args.trade_date:
        result = {"status": "WRONG_LOCAL_DATE", "local_time": now.isoformat(), "orders_sent": False}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    stage = resolve_stage(args.stage, now)
    plan_path = ROOT / "reports" / args.asof / "tomorrow_plan" / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"PLAN_MISSING:{plan_path}")
    plan = json.loads(plan_path.read_text())
    output = plan_path.parent

    if stage == "morning":
        result = {
            "status": "USER_ACTION_REQUIRED_DOMESTIC_SELL_PREVIEW",
            "local_time": now.isoformat(),
            "positions": plan.get("domestic_liquidation_preview", []),
            "next": "체결 완료 후 야간 프리플라이트 실행",
            "orders_sent": False,
        }
        (output / "preflight_morning.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    cfg = json.loads((ROOT / "automation_config.json").read_text())
    broker = KISReadOnly(Path(cfg["credential_file"]), ROOT / "data" / "automation" / "private")
    try:
        entries = night_preflight(plan, broker, reviewed=args.reviewed, now=now)
    finally:
        broker.client.close()

    result = {
        "status": plan["status"],
        "local_time": now.isoformat(),
        "plan_reviewed": args.reviewed,
        "order_count": len(entries),
        "preflight_entries": entries,
        "orders_sent": False,
    }
    (output / "preflight_night.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
