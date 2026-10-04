"""공시 감지와 계좌 조회 및 주문 계획 준비. 실주문 전환은 아직 비활성이다."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from guru_tracker.kis_readonly import KISReadOnly, number
from guru_tracker.study import Fetcher, ROOT, normalize_holdings


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=20)
        # 이 외장 볼륨의 rollback journal은 SQLITE_READONLY_DBMOVED를 발생시킨다.
        # 로컬 단일 호스트 WAL과 FULL 동기화로 기록을 보존한다. 네트워크 공유는 지원하지 않는다.
        if self.db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
            self.db.close()
            raise RuntimeError("WAL_STORAGE_REQUIRED")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS filings (accession TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS intents (id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL)")
        self.db.commit()

    def record_filing(self, accession: str, state: str, payload: dict) -> bool:
        with self.db:
            result = self.db.execute("INSERT OR IGNORE INTO filings VALUES (?, ?, ?)", (accession, state, json.dumps(payload)))
        return result.rowcount == 1

    def record_intent(self, intent: dict) -> bool:
        with self.db:
            result = self.db.execute("INSERT OR IGNORE INTO intents VALUES (?, ?, ?)",
                                     (intent["id"], "PREVIEW_ONLY", json.dumps(intent)))
        return result.rowcount == 1


def pending_decisions(cfg: dict) -> list[str]:
    pending = []
    for key in ("max_position_pct", "allocation", "existing_strategy_policy", "exit_policy", "account_suffix"):
        if cfg.get(key) is None:
            pending.append(key)
    if cfg.get("budget_usd") is None and cfg.get("capital_scope") != "entire_connected_kis_account":
        pending.append("budget_usd")
    if not cfg.get("entry_policy_confirmed"):
        pending.append("entry_policy")
    return pending


def poll(cfg: dict, ledger: Ledger, fetch: Fetcher) -> dict:
    data = fetch.get(f'https://data.sec.gov/submissions/CIK{cfg["manager_cik"]}.json', refresh=True)
    filings = pd.DataFrame(data["filings"]["recent"])
    filings = filings.loc[filings.form.isin(["13F-HR", "13F-HR/A"])].copy()
    now = pd.Timestamp.now(tz="UTC")
    filings = filings.loc[pd.to_datetime(filings.acceptanceDateTime, utc=True) <= now]
    if filings.empty:
        raise RuntimeError("NO_PUBLIC_FILINGS")
    first_run = ledger.db.execute("SELECT count(*) FROM filings").fetchone()[0] == 0
    def record_disclosures():
        new = []
        for filing in filings.sort_values("acceptanceDateTime").to_dict("records"):
            state = "BASELINE_NO_ORDER" if first_run else "DISCOVERED"
            if ledger.record_filing(filing["accessionNumber"], state, filing):
                new.append(filing["accessionNumber"])
        return new
    latest_period = filings.reportDate.max()
    current = filings.loc[filings.reportDate.eq(latest_period)].sort_values("acceptanceDateTime").iloc[-1]
    if current.form == "13F-HR/A":
        return {"status": "AMENDMENT_RECONSTRUCTION_REQUIRED", "accession": current.accessionNumber,
                "new_filings": record_disclosures(), "candidates": [], "orders_sent": False}
    raw = fetch.get(f'https://13f.info/data/13f/{current.accessionNumber.replace("-", "")}')
    holdings = normalize_holdings(raw["data"])
    selected = holdings.loc[holdings.weight_pct >= cfg["minimum_weight_pct"]].copy()
    unknown = [r.symbol for r in selected.itertuples() if cfg["verified_instruments"].get(r.cusip, {}).get("symbol") != r.symbol]
    previous_filings = filings.loc[(filings.reportDate < latest_period) & filings.form.eq("13F-HR")].sort_values("reportDate")
    comparison = {"status": "NO_PREVIOUS_FILING", "changes": []}
    if not previous_filings.empty:
        prior = previous_filings.iloc[-1]
        prior_amended = ((filings.reportDate == prior.reportDate) & filings.form.eq("13F-HR/A")).any()
        if prior_amended:
            comparison = {"status": "PREVIOUS_AMENDMENT_REQUIRES_REVIEW", "changes": []}
        else:
            prior_raw = fetch.get(f'https://13f.info/data/13f/{prior.accessionNumber.replace("-", "")}')
            prior_holdings = normalize_holdings(prior_raw["data"])
            old = {r.cusip: r for r in prior_holdings.itertuples()}
            new_rows = {r.cusip: r for r in holdings.itertuples()}
            keys = set(selected.cusip) | set(prior_holdings.loc[prior_holdings.weight_pct.ge(cfg["minimum_weight_pct"]), "cusip"])
            changes = []
            for cusip in sorted(keys):
                before = old[cusip].weight_pct if cusip in old else 0.
                after = new_rows[cusip].weight_pct if cusip in new_rows else 0.
                before_qualified, after_qualified = before >= cfg["minimum_weight_pct"], after >= cfg["minimum_weight_pct"]
                changes.append({"cusip": cusip, "symbol": (new_rows.get(cusip) or old[cusip]).symbol,
                                "previous_weight_pct": before, "current_weight_pct": after,
                                "weight_change_pp": after - before,
                                "candidate_change": "ENTERED" if not before_qualified else "RETAINED" if after_qualified else "EXITED"})
            comparison = {"status": "PUBLIC_FALLBACK_COMPARISON", "previous_report_date": prior.reportDate,
                          "previous_accession": prior.accessionNumber, "changes": changes,
                          "note": "비중 변화는 주가 변동과 수량 변동을 함께 반영하며 실제 매매의 증거로 단정하지 않는다."}
    # 보유표 수집 실패를 이미 처리한 신규 공시로 기록하지 않는다.
    new = record_disclosures()
    return {
        "status": "BASELINE_NO_ORDER" if first_run else "MONITOR_ONLY", "observed_at": now.isoformat(),
        "accession": current.accessionNumber, "report_date": current.reportDate,
        "accepted_at": current.acceptanceDateTime, "new_filings": new,
        "candidates": selected[["symbol", "issuer", "cusip", "weight_pct"]].to_dict("records"),
        "unverified_symbols": unknown, "source": "SEC_METADATA_AND_13F_INFO_FALLBACK",
        "execution_source_verified": False, "pending_decisions": pending_decisions(cfg),
        "quarter_comparison": comparison,
        "ruleset": cfg.get("active_ruleset", "legacy_5pct"),
        "candidates_role": "LEGACY_REFERENCE_ONLY_NOT_ENTRY_SIGNALS" if cfg.get("active_ruleset") == "multi_guru_draft" else "RESEARCH_CANDIDATES",
        "orders_sent": False,
    }


def main():
    parser = argparse.ArgumentParser(description="구루 자동투자 전환 준비: 감시·조회·계획")
    parser.add_argument("command", choices=["status", "poll", "broker-check"])
    args = parser.parse_args()
    cfg = json.loads((ROOT / "automation_config.json").read_text())
    if cfg.get("mode") != "observe":
        raise SystemExit("LIVE_EXECUTION_NOT_ENABLED: 투자 조건과 실주문 통합 검증이 필요합니다.")
    state = ROOT / "data" / "automation"
    state.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        result = {"mode": cfg["mode"], "pending_decisions": pending_decisions(cfg),
                  "strategy_status": cfg.get("strategy_status"),
                  "local_order_scheduler_installed": False, "live_orders_enabled": False,
                  "monitor_schedule": "Codex heartbeat에서 별도 관리", "orders_sent": False}
    elif args.command == "broker-check":
        broker = KISReadOnly(Path(cfg["credential_file"]), state / "private")
        try:
            result = broker.snapshot(cfg["verified_instruments"])
            path = state / "broker_snapshot.json"
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
            result = {"status": "BROKER_READ_OK", "account_suffix": result["account_suffix"],
                      "holding_count": len(result["holdings"]), "pending_order_count": result["pending_order_count"],
                      "quote_symbols": sorted(result["quotes"]), "orders_sent": False}
        finally:
            broker.client.close()
    else:
        ledger = Ledger(state / "ledger.sqlite3")
        fetch = Fetcher(ROOT / "data")
        try:
            result = poll(cfg, ledger, fetch)
            (state / "latest_disclosure.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            (state / "source_attempts.json").write_text(json.dumps(fetch.attempts, ensure_ascii=False, indent=2))
            ledger.db.close()
            fetch.client.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
