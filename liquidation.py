"""승인된 두 국내 보유종목 청산. 호가 지정가, 매도만, 불명확한 응답 재주문 금지."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

from guru_tracker.kis_readonly import BASE, KISReadOnly, number
from guru_tracker.stop_legacy import LABELS
from guru_tracker.study import ROOT

STATE = ROOT / "data" / "automation"


def time_allowed(cfg: dict, now: datetime) -> bool:
    local = now.astimezone(ZoneInfo("Asia/Seoul"))
    return (cfg["not_before"] <= local.date().isoformat() <= cfg["expires_on"]
            and local.weekday() < 5
            and cfg["first_order_time_kst"] <= local.strftime("%H:%M") <= cfg["last_order_time_kst"])


def legacy_is_stopped() -> bool:
    uid = os.getuid()
    disabled = subprocess.run(["launchctl", "print-disabled", f"gui/{uid}"], capture_output=True, text=True)
    if disabled.returncode:
        return False
    for label in LABELS:
        if not any(f'"{label}" => {value}' in disabled.stdout for value in ("true", "disabled")):
            return False
        running = subprocess.run(["launchctl", "print", f"gui/{uid}/{label}"], capture_output=True, text=True)
        if running.returncode == 0:
            return False
    return True


def pending_domestic(broker: KISReadOnly) -> list[dict]:
    rows, cursor, seen = [], ("", ""), set()
    for page in range(20):
        data, cont = broker.get("/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl", "TTTC8036R", {
            "CANO": broker.account, "ACNT_PRDT_CD": broker.product,
            "CTX_AREA_FK100": cursor[0], "CTX_AREA_NK100": cursor[1], "INQR_DVSN_1": "0", "INQR_DVSN_2": "0",
        }, "N" if page else "")
        if not isinstance(data.get("output"), list):
            raise RuntimeError("PENDING_ORDERS_UNKNOWN")
        rows.extend(data["output"])
        if cont not in {"M", "F"}:
            return rows
        cursor = (data.get("ctx_area_fk100", "").strip(), data.get("ctx_area_nk100", "").strip())
        if cursor == ("", "") or cursor in seen:
            raise RuntimeError("PENDING_ORDERS_INCOMPLETE")
        seen.add(cursor)
    raise RuntimeError("PENDING_ORDERS_PAGE_LIMIT")


def submit_once(broker: KISReadOnly, db: sqlite3.Connection, symbol: str, qty: int, price: int, date: str) -> str:
    identity = f"{broker.fingerprint}:{date}:{symbol}:SELL"
    if qty <= 0 or price <= 0:
        raise ValueError("INVALID_ORDER")
    # 네트워크 호출 전에 영속 기록한다. 응답 유실 시 자동 재전송하지 않는다.
    with db:
        inserted = db.execute("INSERT OR IGNORE INTO liquidation_orders VALUES (?, ?, ?, ?, ?, 'SUBMITTING', '')",
                              (identity, symbol, date, qty, price)).rowcount
    if not inserted:
        return "ALREADY_SUBMITTED_TODAY"
    body = {"CANO": broker.account, "ACNT_PRDT_CD": broker.product, "PDNO": symbol,
            "ORD_DVSN": "00", "ORD_QTY": str(qty), "ORD_UNPR": str(price),
            "EXCG_ID_DVSN_CD": "KRX", "SLL_TYPE": "01", "CNDT_PRIC": ""}
    try:
        response = broker.client.post(BASE + "/uapi/domestic-stock/v1/trading/order-cash", json=body, headers={
            "authorization": "Bearer " + broker.token(), "appkey": broker.key, "appsecret": broker.secret,
            "tr_id": "TTTC0011U", "custtype": "P", "content-type": "application/json; charset=utf-8",
        })
        if response.status_code != 200:
            raise RuntimeError("ORDER_ACK_UNKNOWN")
        data = response.json()
        if data.get("rt_cd") != "0":
            state, order_id = "REJECTED", str(data.get("msg_cd", "UNKNOWN"))
        else:
            output = data.get("output", {})
            order_id = output.get("ODNO") or output.get("odno")
            if not order_id:
                raise RuntimeError("ORDER_ACK_NO_ID")
            state = "ACCEPTED_NOT_FILLED"
    except (httpx.HTTPError, ValueError, RuntimeError):
        state, order_id = "UNKNOWN_REQUIRES_RECONCILIATION", ""
    with db:
        db.execute("UPDATE liquidation_orders SET state=?, broker_order_id=? WHERE id=?", (state, order_id, identity))
    return state


def revise_pending_to_best_bid(cfg: dict, broker: KISReadOnly, db: sqlite3.Connection, now: datetime) -> dict:
    if not cfg["enabled"]:
        return {"status": "DISABLED", "orders_sent": False}
    if broker.fingerprint != cfg["account_fingerprint"] or not time_allowed(cfg, now):
        raise RuntimeError("REVISION_NOT_AUTHORIZED_NOW")
    rows = [row for row in pending_domestic(broker)
            if row.get("pdno") in cfg["authorized_max_quantities"]]
    if len(rows) != 1:
        raise RuntimeError("EXACTLY_ONE_AUTHORIZED_PENDING_ORDER_REQUIRED")
    row = rows[0]
    symbol, order_no = row["pdno"], row["odno"]
    remaining = number(row.get("psbl_qty"))
    if remaining <= 0 or remaining != remaining.to_integral_value():
        raise RuntimeError("PENDING_REMAINING_QUANTITY_INVALID")
    quote, _ = broker.get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", "FHKST01010200",
                          {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol})
    output = quote.get("output1", {})
    bid, stamp = number(output.get("bidp1")), str(output.get("aspr_acpt_hour", ""))
    local = now.astimezone(ZoneInfo("Asia/Seoul"))
    try:
        quoted_at = local.replace(hour=int(stamp[:2]), minute=int(stamp[2:4]), second=int(stamp[4:6]), microsecond=0)
    except (ValueError, TypeError):
        raise RuntimeError("QUOTE_TIMESTAMP_UNKNOWN") from None
    # aspr_acpt_hour는 API 응답 시각이 아니라 마지막 호가 변경 시각이다. 거래가 뜸한
    # 종목의 기존 지정가를 현재 매수 1호가로 정정할 때만 10분까지 허용한다.
    if not 0 <= (datetime.now(ZoneInfo("Asia/Seoul")) - quoted_at).total_seconds() <= 600 or bid <= 0:
        raise RuntimeError("QUOTE_STALE_OR_INVALID")
    identity = f"{broker.fingerprint}:{local:%Y%m%d}:{order_no}:{int(bid)}:REVISE"
    with db:
        inserted = db.execute(
            "INSERT OR IGNORE INTO liquidation_revisions VALUES (?, ?, ?, ?, ?, 'SUBMITTING', '')",
            (identity, symbol, order_no, int(remaining), int(bid))).rowcount
    if not inserted:
        return {"status": "REVISION_ALREADY_SUBMITTED", "orders_sent": False}
    body = {"CANO": broker.account, "ACNT_PRDT_CD": broker.product,
            "KRX_FWDG_ORD_ORGNO": row["ord_gno_brno"], "ORGN_ODNO": order_no,
            "ORD_DVSN": "00", "RVSE_CNCL_DVSN_CD": "01", "ORD_QTY": str(int(remaining)),
            "ORD_UNPR": str(int(bid)), "QTY_ALL_ORD_YN": "Y", "EXCG_ID_DVSN_CD": "KRX", "CNDT_PRIC": ""}
    try:
        response = broker.client.post(BASE + "/uapi/domestic-stock/v1/trading/order-rvsecncl", json=body, headers={
            "authorization": "Bearer " + broker.token(), "appkey": broker.key, "appsecret": broker.secret,
            "tr_id": "TTTC0013U", "custtype": "P", "content-type": "application/json; charset=utf-8",
        })
        if response.status_code != 200:
            raise RuntimeError("REVISION_ACK_UNKNOWN")
        data = response.json()
        if data.get("rt_cd") != "0":
            state, revised_order_id = "REJECTED", str(data.get("msg_cd", "UNKNOWN"))
        else:
            revised_order_id = data.get("output", {}).get("ODNO") or data.get("output", {}).get("odno")
            if not revised_order_id:
                raise RuntimeError("REVISION_ACK_NO_ID")
            state = "ACCEPTED_NOT_FILLED"
    except (httpx.HTTPError, ValueError, RuntimeError):
        state, revised_order_id = "UNKNOWN_REQUIRES_RECONCILIATION", ""
    with db:
        db.execute("UPDATE liquidation_revisions SET state=?, broker_order_id=? WHERE id=?",
                   (state, revised_order_id, identity))
    return {"status": "LIQUIDATION_ORDER_REVISED", "symbol": symbol,
            "remaining_quantity": int(remaining), "limit_price_krw": int(bid),
            "revision_state": state, "orders_sent": state != "REJECTED"}


def tick(cfg: dict, broker: KISReadOnly, db: sqlite3.Connection, *, execute: bool, now: datetime) -> dict:
    # enabled는 실주문 스위치다. 읽기 전용 프리뷰까지 막으면 활성화 전에 검증할 수 없다.
    if not cfg["enabled"] and execute:
        return {"status": "DISABLED", "orders_sent": False}
    if broker.fingerprint != cfg["account_fingerprint"]:
        raise RuntimeError("ACCOUNT_MISMATCH")
    if not time_allowed(cfg, now):
        return {"status": "WAIT_KRX_SESSION", "not_before": cfg["not_before"], "orders_sent": False}
    if not legacy_is_stopped():
        raise RuntimeError("LEGACY_STRATEGY_NOT_STOPPED")
    unresolved = db.execute("SELECT count(*) FROM liquidation_orders WHERE state IN ('SUBMITTING','UNKNOWN_REQUIRES_RECONCILIATION')").fetchone()[0]
    if unresolved:
        return {"status": "UNKNOWN_ORDER_REQUIRES_RECONCILIATION", "orders_sent": False}
    today = now.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    calendar_path = STATE / f"krx_calendar_{today}.json"
    if calendar_path.exists():
        calendar = json.loads(calendar_path.read_text())
    else:
        calendar, _ = broker.get("/uapi/domestic-stock/v1/quotations/chk-holiday", "CTCA0903R",
                                 {"BASS_DT": today, "CTX_AREA_FK": "", "CTX_AREA_NK": ""})
        calendar_path.write_text(json.dumps(calendar))
    day = next((r for r in calendar.get("output", []) if r.get("bass_dt") == today), None)
    if not day or day.get("opnd_yn") != "Y":
        return {"status": "KRX_CLOSED_OR_UNVERIFIED", "orders_sent": False}
    balance = broker.domestic_snapshot()
    if pending_domestic(broker):
        return {"status": "WAIT_EXISTING_ORDERS", "orders_sent": False}
    authorized = cfg["authorized_max_quantities"]
    holdings = [r for r in balance["holdings"] if r["symbol"] in authorized]
    if not holdings:
        return {"status": "DOMESTIC_LIQUIDATION_COMPLETE", "orders_sent": False,
                "next_step": "VERIFY_SETTLED_OR_INTEGRATED_OVERSEAS_BUYING_POWER"}
    results = []
    for row in holdings:
        symbol = row["symbol"]
        quantity = number(row["quantity"])
        if quantity != quantity.to_integral_value() or quantity > authorized[symbol]:
            raise RuntimeError("POSITION_CHANGED_BEYOND_AUTHORIZATION")
        sellable = number(row["sellable_quantity"])
        if sellable != quantity:
            raise RuntimeError("POSITION_NOT_FULLY_SELLABLE")
        quote, _ = broker.get("/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn", "FHKST01010200",
                              {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol})
        output = quote.get("output1", {})
        bid = number(output.get("bidp1"))
        stamp = str(output.get("aspr_acpt_hour", ""))
        local = now.astimezone(ZoneInfo("Asia/Seoul"))
        try:
            quoted_at = local.replace(hour=int(stamp[:2]), minute=int(stamp[2:4]), second=int(stamp[4:6]), microsecond=0)
        except (ValueError, TypeError):
            raise RuntimeError("QUOTE_TIMESTAMP_UNKNOWN") from None
        if not 0 <= (datetime.now(ZoneInfo("Asia/Seoul")) - quoted_at).total_seconds() <= 90 or bid <= 0 or bid != bid.to_integral_value():
            raise RuntimeError("QUOTE_STALE_OR_INVALID")
        status = submit_once(broker, db, symbol, int(quantity), int(bid), today) if execute else "PREVIEW_ONLY"
        results.append({"symbol": symbol, "quantity": int(quantity), "limit_price_krw": int(bid), "status": status})
        if status == "UNKNOWN_REQUIRES_RECONCILIATION":
            break
    return {"status": "LIQUIDATION_ORDERS", "orders": results,
            "orders_sent": any(r["status"] in {"ACCEPTED_NOT_FILLED", "UNKNOWN_REQUIRES_RECONCILIATION"} for r in results)}


def main():
    parser = argparse.ArgumentParser(description="승인된 기존 국내주식 청산")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reprice-pending", action="store_true")
    args = parser.parse_args()
    STATE.mkdir(parents=True, exist_ok=True)
    with (STATE / "liquidation.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        cfg = json.loads((ROOT / "liquidation_config.json").read_text())
        common = json.loads((ROOT / "automation_config.json").read_text())
        broker = KISReadOnly(Path(common["credential_file"]), STATE / "private")
        db = sqlite3.connect(STATE / "liquidation.sqlite3")
        db.execute("CREATE TABLE IF NOT EXISTS liquidation_orders (id TEXT PRIMARY KEY, symbol TEXT, date TEXT, quantity INTEGER, price INTEGER, state TEXT, broker_order_id TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS liquidation_revisions (id TEXT PRIMARY KEY, symbol TEXT, original_order_id TEXT, quantity INTEGER, price INTEGER, state TEXT, broker_order_id TEXT)")
        try:
            now = datetime.now(ZoneInfo("Asia/Seoul"))
            result = (revise_pending_to_best_bid(cfg, broker, db, now) if args.reprice_pending
                      else tick(cfg, broker, db, execute=args.execute, now=now))
        except Exception as exc:
            result = {"status": "BLOCKED", "reason": str(exc), "orders_sent": "CHECK_LEDGER"}
        finally:
            db.close()
            broker.client.close()
        result["timestamp"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
        (STATE / "liquidation_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
