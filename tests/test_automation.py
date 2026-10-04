from datetime import datetime, timezone

import httpx
import pytest

from guru_tracker.automation import Ledger, poll
from guru_tracker.kis_readonly import KISReadOnly, number


def plan_inputs():
    candidates = [{"cusip": "A", "symbol": "AAA", "weight_pct": 20}, {"cusip": "B", "symbol": "BBB", "weight_pct": 10}]
    cfg = {"mode": "observe", "budget_usd": 1000, "max_position_pct": 35, "allocation": "equal_weight_capped",
           "existing_strategy_policy": "review", "exit_policy": "below_5pct", "account_suffix": "1234",
           "entry_policy_confirmed": True, "minimum_weight_pct": 5, "entry_offsets_sessions": [0],
           "verified_instruments": {"A": {"symbol": "AAA", "exchange": "NASD"}, "B": {"symbol": "BBB", "exchange": "NYSE"}}}
    snapshot = {"source": "KIS_LIVE_READ_ONLY", "observed_at": datetime.now(timezone.utc).isoformat(),
                "account_suffix": "1234", "account_fingerprint": "test-only", "pending_order_count": 0,
                "available_usd": 1000, "holdings": [], "quotes": {"AAA": {"price": 100, "exchange": "NASD"}, "BBB": {"price": 10, "exchange": "NYSE"}}}
    return candidates, snapshot, cfg



@pytest.mark.parametrize("value", ["NaN", "Infinity", None, ""])
def test_invalid_broker_number(value):
    with pytest.raises(ValueError):
        number(value)


def test_readonly_rejects_order_endpoint_before_auth():
    broker = KISReadOnly.__new__(KISReadOnly)
    with pytest.raises(ValueError, match="READ_ONLY_ENDPOINT"):
        broker.get("/uapi/overseas-stock/v1/trading/order", "TTTT1002U", {})


def test_pagination_incomplete_is_not_success():
    broker = KISReadOnly.__new__(KISReadOnly)
    broker.account, broker.product = "00000000", "01"
    broker.get = lambda *args: ({"output": []}, "M")
    with pytest.raises(RuntimeError, match="PAGINATION_INCOMPLETE"):
        broker.pages("irrelevant", "test", {}, "output")


def test_find_us_instrument_treats_not_listed_here_as_normal(monkeypatch):
    """NASD에서만 상장된 종목은 NYSE/AMEX 조회가 APBN0024(조회된 데이터가 없습니다)로
    응답한다. 이는 실패가 아니라 다른 두 거래소를 마저 확인해야 하는 정상 결과다."""
    broker = KISReadOnly.__new__(KISReadOnly)
    broker.key, broker.secret = "TEST", "TEST"
    broker.token = lambda: "TEST"
    calls = []

    def handler(request):
        product_type = httpx.QueryParams(request.url.query)["PRDT_TYPE_CD"]
        calls.append(product_type)
        if product_type == "512":
            return httpx.Response(200, json={"rt_cd": "0", "output": {
                "std_pdno": "US0231351067", "prdt_name": "아마존닷컴", "ovrs_item_name": ""}})
        return httpx.Response(200, json={"rt_cd": "7", "msg_cd": "APBN0024",
                                         "msg1": "조회된 데이터가 없습니다.(해외주식기본)"})

    broker.client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("guru_tracker.kis_readonly.time.sleep", lambda x: None)
    try:
        result = broker.find_us_instrument("AMZN", "023135106")
        assert result == {"symbol": "AMZN", "cusip": "023135106", "exchange": "NASD",
                          "product_type": "512", "standard_code": "US0231351067", "name": "아마존닷컴"}
        assert calls == ["512", "513", "529"]
    finally:
        broker.client.close()


def test_find_us_instrument_genuine_failure_still_blocks(monkeypatch):
    """APBN0024가 아닌 실제 오류(예: 인증/서버 오류)는 여전히 조회 불능으로 막는다."""
    broker = KISReadOnly.__new__(KISReadOnly)
    broker.key, broker.secret = "TEST", "TEST"
    broker.token = lambda: "TEST"

    def handler(request):
        product_type = httpx.QueryParams(request.url.query)["PRDT_TYPE_CD"]
        if product_type == "512":
            return httpx.Response(200, json={"rt_cd": "0", "output": {
                "std_pdno": "US0231351067", "prdt_name": "아마존닷컴"}})
        return httpx.Response(200, json={"rt_cd": "9", "msg_cd": "SOME_OTHER_ERROR", "msg1": "일시 오류"})

    broker.client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("guru_tracker.kis_readonly.time.sleep", lambda x: None)
    try:
        with pytest.raises(RuntimeError, match="KIS_INSTRUMENT_LOOKUP_UNAVAILABLE"):
            broker.find_us_instrument("AMZN", "023135106")
    finally:
        broker.client.close()


def test_read_http_retries_do_not_include_order_posts(monkeypatch):
    broker = KISReadOnly.__new__(KISReadOnly)
    broker.key, broker.secret = "TEST", "TEST"
    broker.token = lambda: "TEST"
    calls = []
    def handler(request):
        calls.append(request.method)
        return httpx.Response(429 if len(calls) < 3 else 200, json={"rt_cd": "0", "output": {}})
    broker.client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("guru_tracker.kis_readonly.time.sleep", lambda x: None)
    try:
        broker.get("/uapi/overseas-price/v1/quotations/price", "HHDFS00000300", {})
        assert calls == ["GET", "GET", "GET"]
    finally:
        broker.client.close()


class FilingFetcher:
    def __init__(self):
        self.fail = False
        self.accessions = ["0000000000-25-000001"]

    def get(self, url, **kwargs):
        if "submissions" in url:
            n = len(self.accessions)
            return {"filings": {"recent": {"accessionNumber": self.accessions,
                    "form": ["13F-HR"] * n, "reportDate": ["2024-12-31", "2025-03-31"][:n],
                    "acceptanceDateTime": ["2025-02-14T22:00:00Z", "2025-05-14T22:00:00Z"][:n]}}}
        if self.fail:
            raise RuntimeError("HOLDINGS_FAILED")
        return {"data": [["AAA", "TEST", "COM", "A", 100, 100, 10, None, None]]}


def test_poll_baseline_new_filing_and_failed_fetch_not_acknowledged(tmp_path):
    _, _, cfg = plan_inputs()
    cfg["manager_cik"] = "0000000000"
    ledger = Ledger(tmp_path / "poll.sqlite3")
    fetch = FilingFetcher()
    try:
        first = poll(cfg, ledger, fetch)
        assert first["status"] == "BASELINE_NO_ORDER"
        assert first["orders_sent"] is False
        assert poll(cfg, ledger, fetch)["new_filings"] == []
        fetch.accessions.append("0000000000-25-000002")
        fetch.fail = True
        with pytest.raises(RuntimeError, match="HOLDINGS_FAILED"):
            poll(cfg, ledger, fetch)
        assert ledger.db.execute("SELECT count(*) FROM filings").fetchone()[0] == 1
        fetch.fail = False
        assert poll(cfg, ledger, fetch)["new_filings"] == ["0000000000-25-000002"]
    finally:
        ledger.db.close()


def test_ledger_reopens_without_losing_records(tmp_path):
    path = tmp_path / "durable.sqlite3"
    ledger = Ledger(path)
    assert ledger.record_filing("one", "BASELINE_NO_ORDER", {})
    ledger.db.close()
    again = Ledger(path)
    try:
        assert not again.record_filing("one", "BASELINE_NO_ORDER", {})
        assert again.record_filing("two", "DISCOVERED", {})
        assert again.db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        again.db.close()
