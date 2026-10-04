"""다음 거래일을 위한 리밸런싱 초안. 조회만 하며 주문 API를 호출하지 않는다."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import pandas as pd

from guru_tracker.kis_readonly import KISReadOnly, number
from guru_tracker.study import ROOT

INSTRUMENT_CACHE_MAX_HOURS = 24
TARGET_WEIGHT_PCT = Decimal("12.5")
TOLERANCE_PCT = Decimal("25")   # ±25% of target → band 9.375%~15.625%


def quantity_for(equity_usd, target_pct, price) -> int:
    equity, target, quote = number(equity_usd), number(target_pct), number(price)
    if equity <= 0 or quote <= 0 or not 0 < target < 100:
        raise ValueError("INVALID_ALLOCATION_INPUT")
    return int((equity * target / 100 / quote).to_integral_value(rounding=ROUND_DOWN))


def full_investment_quantities(equity_usd, targets: dict[str, float], quotes: dict) -> dict[str, int]:
    """목표비중 내림 수량에서 시작해 차입 없이 잔액을 가능한 한 투자한다."""
    equity = number(equity_usd)
    if equity <= 0 or not targets or abs(sum(map(float, targets.values())) - 100) > 1e-9:
        raise ValueError("FULL_INVESTMENT_TARGETS_MUST_SUM_TO_100")
    prices = {symbol: number(quotes[symbol]["price"]) for symbol in targets}
    if any(price <= 0 for price in prices.values()):
        raise ValueError("INVALID_FULL_INVESTMENT_PRICE")
    quantities = {symbol: quantity_for(equity, target, prices[symbol])
                  for symbol, target in targets.items()}
    remaining = equity - sum(prices[symbol] * qty for symbol, qty in quantities.items())
    while True:
        affordable = [symbol for symbol, price in prices.items() if price <= remaining]
        if not affordable:
            break
        symbol = min(affordable, key=lambda item: (
            prices[item] * (quantities[item] + 1) / equity * 100 - number(targets[item]), item))
        quantities[symbol] += 1
        remaining -= prices[symbol]
    return quantities


def build_rebalance_plan(
    target_entries: list[dict],
    current_holdings: list[dict],
    equity_usd: Decimal,
    quotes: dict[str, dict],
) -> list[dict]:
    """12.5% 동일비중 목표 대비 현재 보유를 비교해 매매 계획을 생성한다.

    신규 편입·편출은 허용 범위 우선 비교 없이 바로 처리한다.
    기존 종목은 현재 비중이 9.375%~15.625% 범위 안이면 매매를 강제하지 않는다.
    """
    if equity_usd <= 0:
        raise ValueError("INVALID_EQUITY")
    lower_band = TARGET_WEIGHT_PCT * (1 - TOLERANCE_PCT / 100)   # 9.375
    upper_band = TARGET_WEIGHT_PCT * (1 + TOLERANCE_PCT / 100)   # 15.625

    target_symbols = {e["symbol"]: e for e in target_entries}
    current_by_symbol = {h["symbol"]: h for h in current_holdings}
    rows: list[dict] = []

    for entry in target_entries:
        symbol = entry["symbol"]
        cusip = entry["cusip"]
        if symbol not in quotes:
            raise ValueError(f"QUOTE_MISSING:{symbol}")
        price = number(quotes[symbol]["price"])
        if price <= 0:
            raise ValueError(f"INVALID_PRICE:{symbol}")
        target_qty = quantity_for(equity_usd, TARGET_WEIGHT_PCT, price)

        if symbol not in current_by_symbol:
            rows.append({
                "symbol": symbol, "cusip": cusip,
                "action": "NEW_ENTRY",
                "current_weight_pct": 0.0,
                "target_weight_pct": float(TARGET_WEIGHT_PCT),
                "current_qty": 0, "target_qty": target_qty, "delta_qty": target_qty,
                "reference_price_usd": str(price),
                "reason": "not_in_current_holdings",
            })
            continue

        holding = current_by_symbol[symbol]
        current_qty = int(number(holding["quantity"]))
        current_value = number(holding["market_value_usd"])
        current_weight = current_value / equity_usd * 100

        if current_weight < lower_band:
            delta = max(0, target_qty - current_qty)
            rows.append({
                "symbol": symbol, "cusip": cusip,
                "action": "BUY_REBALANCE",
                "current_weight_pct": float(current_weight.quantize(Decimal("0.0001"))),
                "target_weight_pct": float(TARGET_WEIGHT_PCT),
                "current_qty": current_qty, "target_qty": target_qty, "delta_qty": delta,
                "reference_price_usd": str(price),
                "reason": f"below_band_lower={float(lower_band):.4f}pct",
            })
        elif current_weight > upper_band:
            delta = max(0, current_qty - target_qty)
            rows.append({
                "symbol": symbol, "cusip": cusip,
                "action": "SELL_REBALANCE",
                "current_weight_pct": float(current_weight.quantize(Decimal("0.0001"))),
                "target_weight_pct": float(TARGET_WEIGHT_PCT),
                "current_qty": current_qty, "target_qty": target_qty, "delta_qty": delta,
                "reference_price_usd": str(price),
                "reason": f"above_band_upper={float(upper_band):.4f}pct",
            })
        else:
            rows.append({
                "symbol": symbol, "cusip": cusip,
                "action": "HOLD",
                "current_weight_pct": float(current_weight.quantize(Decimal("0.0001"))),
                "target_weight_pct": float(TARGET_WEIGHT_PCT),
                "current_qty": current_qty, "target_qty": target_qty, "delta_qty": 0,
                "reference_price_usd": str(price),
                "reason": f"within_band_{float(lower_band):.4f}pct_{float(upper_band):.4f}pct",
            })

    for symbol, holding in current_by_symbol.items():
        if symbol in target_symbols:
            continue
        current_qty = int(number(holding["quantity"]))
        current_value = number(holding["market_value_usd"])
        current_weight = current_value / equity_usd * 100
        price_str = quotes[symbol]["price"] if symbol in quotes else "PRICE_NOT_PREFETCHED"
        rows.append({
            "symbol": symbol, "cusip": "UNKNOWN",
            "action": "EXIT_SELL",
            "current_weight_pct": float(current_weight.quantize(Decimal("0.0001"))),
            "target_weight_pct": 0.0,
            "current_qty": current_qty, "target_qty": 0, "delta_qty": current_qty,
            "reference_price_usd": price_str,
            "reason": "not_in_target_portfolio",
        })

    return rows


def safe_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def cached_instrument(plan_path: Path, symbol: str, cusip: str, now: datetime | None = None) -> dict:
    """최근 KIS CUSIP 검증만 재사용한다. 가격·잔고 캐시에는 사용하지 않는다."""
    now = now or datetime.now(timezone.utc)
    if not plan_path.exists():
        raise RuntimeError("RECENT_KIS_INSTRUMENT_CACHE_MISSING")
    plan = json.loads(plan_path.read_text())
    created = datetime.fromisoformat(plan["created_at"])
    age_hours = (now - created.astimezone(timezone.utc)).total_seconds() / 3600
    if age_hours < 0 or age_hours > INSTRUMENT_CACHE_MAX_HOURS:
        raise RuntimeError("RECENT_KIS_INSTRUMENT_CACHE_EXPIRED")
    matches = [item for item in plan.get("verified_instruments", [])
               if item.get("symbol") == symbol and item.get("cusip") == cusip
               and str(item.get("standard_code", "")).startswith("US" + cusip)]
    if len(matches) != 1:
        raise RuntimeError("RECENT_KIS_INSTRUMENT_CACHE_NOT_UNIQUE")
    return {**matches[0], "verification_source": "KIS_EXACT_CUSIP_CACHE_LT_24H"}


def krx_trade_day(broker: KISReadOnly, trade_date: str) -> dict:
    key = trade_date.replace("-", "")
    cache = ROOT / "data" / "automation" / f"krx_calendar_{key}.json"
    if cache.exists():
        data = json.loads(cache.read_text())
    else:
        data, _ = broker.get("/uapi/domestic-stock/v1/quotations/chk-holiday", "CTCA0903R",
                             {"BASS_DT": key, "CTX_AREA_FK": "", "CTX_AREA_NK": ""})
        safe_write(cache, data)
    day = next((r for r in data.get("output", []) if r.get("bass_dt") == key), None)
    if not day:
        raise RuntimeError("KRX_TRADE_DATE_UNVERIFIED")
    return day


def run(asof: str, trade_date: str) -> Path:
    if pd.Timestamp(trade_date).date() <= pd.Timestamp(asof).date():
        raise ValueError("TRADE_DATE_MUST_BE_AFTER_ASOF")

    portfolio_path = ROOT / "reports" / asof / "selection" / "portfolio.csv"
    if not portfolio_path.exists():
        raise FileNotFoundError(f"SELECTION_PORTFOLIO_MISSING:{portfolio_path}")
    portfolio_df = pd.read_csv(portfolio_path)

    multi_meta_path = ROOT / "reports" / asof / "multi_guru" / "metadata.json"
    if not multi_meta_path.exists():
        raise FileNotFoundError(f"MULTI_GURU_METADATA_MISSING:{multi_meta_path}")
    multi_meta = json.loads(multi_meta_path.read_text())
    if multi_meta.get("sample_data_used") is not False:
        raise RuntimeError("MULTI_GURU_DATA_NOT_VERIFIED")

    target_entries = portfolio_df[["symbol", "cusip", "target_weight_pct"]].to_dict("records")
    cfg = json.loads((ROOT / "automation_config.json").read_text())
    broker = KISReadOnly(Path(cfg["credential_file"]), ROOT / "data" / "automation" / "private")
    output = ROOT / "reports" / asof / "tomorrow_plan"
    output.mkdir(parents=True, exist_ok=True)

    try:
        # CUSIP 검증: 목표 종목만 사전 확인. 현재 보유 중인 편출 대상은 실행 시점에 확인.
        instruments: dict[str, dict] = {}
        verified: list[dict] = []
        for entry in target_entries:
            try:
                item = {**broker.find_us_instrument(entry["symbol"], str(entry["cusip"])),
                        "verification_source": "KIS_LIVE_EXACT_CUSIP"}
            except RuntimeError as exc:
                if str(exc) != "KIS_INSTRUMENT_LOOKUP_UNAVAILABLE":
                    raise
                item = cached_instrument(output / "plan.json", entry["symbol"], str(entry["cusip"]))
            verified.append(item)
            instruments[str(entry["cusip"])] = {"symbol": item["symbol"], "exchange": item["exchange"]}

        overseas = broker.snapshot(instruments)
        domestic = broker.domestic_snapshot()
        margin = broker.integrated_margin_snapshot()
        day = krx_trade_day(broker, trade_date)

        if day.get("opnd_yn") != "Y":
            raise RuntimeError("KRX_TRADE_DATE_CLOSED_OR_UNVERIFIED")
        if overseas["account_suffix"] != cfg["account_suffix"] or domestic["account_suffix"] != cfg["account_suffix"]:
            raise RuntimeError("ACCOUNT_MISMATCH")
        if margin["subscription"] != "통합증거금신청":
            raise RuntimeError("INTEGRATED_MARGIN_NOT_SUBSCRIBED")

        fx = number(margin["usd_reference_fx"])
        equity_krw = number(domestic["total_equity_krw"])
        if fx <= 0 or equity_krw <= 0:
            raise RuntimeError("INVALID_ACCOUNT_EQUITY_OR_FX")

        equity_usd = (equity_krw / fx).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        rebalance = build_rebalance_plan(target_entries, overseas["holdings"], equity_usd, overseas["quotes"])

        has_pending = overseas["pending_order_count"] > 0
        has_domestic = bool(domestic["holdings"])
        has_exit = any(r["action"] == "EXIT_SELL" for r in rebalance)
        has_new = any(r["action"] in ("NEW_ENTRY", "BUY_REBALANCE") for r in rebalance)
        orderable_usd = number(margin["usd_integrated_orderable"])
        buy_notional = sum(
            number(r["reference_price_usd"]) * r["delta_qty"]
            for r in rebalance
            if r["action"] in ("NEW_ENTRY", "BUY_REBALANCE") and r["reference_price_usd"] != "PRICE_NOT_PREFETCHED"
        )

        if has_pending:
            status = "PENDING_ORDERS_PRESENT_RECONCILE_FIRST"
        elif has_domestic:
            status = "WAIT_DOMESTIC_LIQUIDATION_AND_RECHECK"
        elif has_exit:
            status = "EXIT_REQUIRED_BEFORE_REBALANCE"
        elif not has_new:
            status = "HOLD_NO_TRADES_REQUIRED"
        elif orderable_usd < buy_notional:
            status = "INSUFFICIENT_VERIFIED_BUYING_POWER"
        else:
            status = "READY_FOR_USER_ORDER_ENTRY_AFTER_LIVE_QUOTE_CHECK"

        plan = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "asof": asof, "trade_date": trade_date,
            "account_suffix": cfg["account_suffix"],
            "sample_data_used": False, "orders_sent": False,
            "status": status,
            "equity_krw": str(equity_krw),
            "reference_fx_krw_per_usd": str(fx),
            "projected_equity_usd": str(equity_usd),
            "current_integrated_orderable_usd": str(orderable_usd),
            "projected_buy_notional_usd": str(buy_notional),
            "krx_trade_date_verified_open": True,
            "pending_overseas_orders": overseas["pending_order_count"],
            "target_portfolio_count": len(target_entries),
            "rebalance_actions": rebalance,
            "domestic_liquidation_preview": [
                {"symbol": r["symbol"], "name": r["name"],
                 "quantity": r["sellable_quantity"],
                 "order_type": "LIMIT_AT_CURRENT_BEST_BID_AFTER_FRESH_QUOTE",
                 "state": "PREVIEW_NOT_ORDER"}
                for r in domestic["holdings"]
            ],
            "verified_instruments": verified,
            "execution_gates": [
                "미체결 해외 주문이 있으면 매매 계획 확정 전 반드시 대조한다.",
                "국내 보유가 있으면 매도 완료 후 재조회한다.",
                "편출 종목(EXIT_SELL)은 기준 가격이 없으면 실행 시점 시세를 직접 확인한다.",
                "09:07 KST 이후 국내 보유 종목 매도 가능 수량과 실시간 호가를 다시 확인한다.",
                "22:20 KST 이후 통합증거금 USD 주문 가능액을 재조회하고 계획 금액 이상일 때만 계속한다.",
                "미국 정규장 실시간 최우선 매도호가가 기준 가격의 103% 이하인 종목만 지정가 검토한다.",
                "주문 응답이 불명확하면 재전송하지 않고 체결·미체결 조회한다.",
                "평균 매입가 대비 -20% 경보는 알림과 재검토만이며 자동 매도하지 않는다.",
            ],
        }
        safe_write(output / "plan.json", plan)
        pd.DataFrame(rebalance).to_csv(output / "rebalance_actions.csv", index=False)
        pd.DataFrame(plan["domestic_liquidation_preview"]).to_csv(
            output / "domestic_sell_preview.csv", index=False)

        action_counts = {a: sum(1 for r in rebalance if r["action"] == a)
                         for a in ("NEW_ENTRY", "BUY_REBALANCE", "SELL_REBALANCE", "HOLD", "EXIT_SELL")}
        lines = [
            "# 다음 거래일 리밸런싱 계획", "",
            f"기준일 {asof}, 예정일 {trade_date}. 실주문이 아닌 사용자 확인용 계획이다.",
            f"계좌 평가액 {int(equity_krw):,}원, 기준환율 {float(fx):,.2f}원/USD,"
            f" 추정 USD {float(equity_usd):,.2f}.",
            f"상태: **{status}**", "",
            "## 목표 포트폴리오 (12.5% 동일비중)", "",
            "| 행동 | 종목 | CUSIP | 현재 비중 | 목표 비중 | 현재 주 | 목표 주 | 변동 | 기준 가격 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        action_order = ["EXIT_SELL", "NEW_ENTRY", "BUY_REBALANCE", "SELL_REBALANCE", "HOLD"]
        for a in action_order:
            for r in (row for row in rebalance if row["action"] == a):
                lines.append(
                    f"| {r['action']} | {r['symbol']} | {r['cusip']}"
                    f" | {r['current_weight_pct']:.2f}% | {r['target_weight_pct']:.1f}%"
                    f" | {r['current_qty']} | {r['target_qty']} | {r['delta_qty']:+d}"
                    f" | ${r['reference_price_usd']} |"
                )
        lines += [
            "",
            f"신규: {action_counts['NEW_ENTRY']}개  매수조정: {action_counts['BUY_REBALANCE']}개  "
            f"매도조정: {action_counts['SELL_REBALANCE']}개  유지: {action_counts['HOLD']}개  "
            f"편출: {action_counts['EXIT_SELL']}개",
            f"매수 예상 금액: USD {float(buy_notional):,.2f}  주문 가능: USD {float(orderable_usd):,.2f}",
        ]
        (output / "checklist.md").write_text("\n".join(lines), encoding="utf-8")
        return output
    finally:
        broker.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="다음 거래일 리밸런싱 초안. 주문 없음.")
    parser.add_argument("--asof", required=True)
    parser.add_argument("--trade-date", required=True)
    args = parser.parse_args()
    print(run(args.asof, args.trade_date))
