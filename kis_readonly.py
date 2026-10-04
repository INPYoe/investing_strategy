"""한국투자증권 해외주식 조회. 주문·취소·환전 API는 이 모듈에 없다."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
from dotenv import dotenv_values

BASE = "https://openapi.koreainvestment.com:9443"


def number(value) -> Decimal:
    if value is None or str(value).strip() == "":
        raise ValueError("MISSING_NUMBER")
    result = Decimal(str(value).replace(",", ""))
    if not result.is_finite():
        raise ValueError("NONFINITE_NUMBER")
    return result


class KISReadOnly:
    def __init__(self, credential_file: Path, private_dir: Path, *, transport=None):
        env = {**dotenv_values(credential_file), **os.environ}
        self.key = env.get("KIS_APP_KEY_LIVE") or env.get("KIS_APP_KEY")
        self.secret = env.get("KIS_APP_SECRET_LIVE") or env.get("KIS_APP_SECRET")
        raw = env.get("KIS_ACCOUNT_NO_LIVE") or env.get("KIS_ACCOUNT_NO") or ""
        parts = raw.split("-", 1)
        self.account = parts[0].strip()
        self.product = env.get("KIS_ACCOUNT_PRODUCT") or (parts[1].strip() if len(parts) == 2 else "01")
        if not self.key or not self.secret or len(self.account) != 8 or not self.account.isdigit():
            raise ValueError("KIS_CREDENTIALS_MISSING_OR_INVALID")
        self.private_dir = private_dir
        self.private_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = hashlib.sha256(f"{self.account}:{self.product}:{self.key}".encode()).hexdigest()[:16]
        self.client = httpx.Client(timeout=30, transport=transport)

    def token(self) -> str:
        path = self.private_dir / f"token_{self.fingerprint}.json"
        if path.exists():
            try:
                cached = json.loads(path.read_text())
                if cached["expires_at"] > time.time() + 300:
                    return cached["token"]
            except (ValueError, KeyError):
                pass
        response = self.client.post(BASE + "/oauth2/tokenP", json={
            "grant_type": "client_credentials", "appkey": self.key, "appsecret": self.secret,
        })
        if response.status_code != 200:
            raise RuntimeError(f"KIS_TOKEN_HTTP_{response.status_code}")
        data = response.json()
        if not data.get("access_token"):
            raise RuntimeError("KIS_TOKEN_REJECTED")
        payload = {"token": data["access_token"], "expires_at": time.time() + int(data.get("expires_in", 86400))}
        # 임시 디렉터리의 공용 토큰 대신 계좌·앱별 전용 파일을 사용한다.
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
        os.chmod(path, 0o600)
        return payload["token"]

    def get(self, endpoint: str, tr_id: str, params: dict, continuation: str = ""):
        allowed = {
            "/uapi/overseas-stock/v1/trading/inquire-balance",
            "/uapi/overseas-stock/v1/trading/inquire-nccs",
            "/uapi/overseas-stock/v1/trading/inquire-psamount",
            "/uapi/overseas-price/v1/quotations/price",
            "/uapi/domestic-stock/v1/trading/inquire-balance",
            "/uapi/domestic-stock/v1/trading/inquire-psbl-rvsecncl",
            "/uapi/domestic-stock/v1/trading/intgr-margin",
            "/uapi/domestic-stock/v1/trading/inquire-daily-ccld",
            "/uapi/domestic-stock/v1/quotations/chk-holiday",
            "/uapi/domestic-stock/v1/quotations/inquire-price",
            "/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn",
            "/uapi/overseas-price/v1/quotations/search-info",
        }
        if endpoint not in allowed:
            raise ValueError("READ_ONLY_ENDPOINT_REQUIRED")
        headers = {"authorization": "Bearer " + self.token(), "appkey": self.key,
                   "appsecret": self.secret, "tr_id": tr_id, "custtype": "P", "tr_cont": continuation}
        for attempt in range(3):
            try:
                time.sleep(.3)
                response = self.client.get(BASE + endpoint, headers=headers, params=params)
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < 2:
                        time.sleep((2, 5)[attempt])
                        continue
                if response.status_code != 200:
                    raise RuntimeError(f"KIS_READ_HTTP_{response.status_code}")
                data = response.json()
                if data.get("rt_cd") != "0":
                    raise RuntimeError(f"KIS_READ_REJECTED:{data.get('msg_cd', 'UNKNOWN')}")
                return data, response.headers.get("tr_cont", "")
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise RuntimeError(f"KIS_READ_TRANSPORT_FAILURE:{type(exc).__name__}") from None
                time.sleep((2, 5)[attempt])
        raise RuntimeError("KIS_READ_FAILED")

    def pages(self, endpoint: str, tr_id: str, params: dict, field: str) -> list[dict]:
        rows = []
        cursor = ("", "")
        seen = set()
        for page in range(20):
            query = {"CANO": self.account, "ACNT_PRDT_CD": self.product, **params,
                     "CTX_AREA_FK200": cursor[0], "CTX_AREA_NK200": cursor[1]}
            data, continuation = self.get(endpoint, tr_id, query, "N" if page else "")
            if field not in data:
                raise RuntimeError("KIS_INCOMPLETE_RESPONSE")
            items = data[field]
            if isinstance(items, dict):
                items = [items]
            if not isinstance(items, list):
                raise RuntimeError("KIS_INVALID_RESPONSE_ROWS")
            rows.extend(items)
            if continuation not in {"M", "F"}:
                return rows
            cursor = (data.get("ctx_area_fk200", "").strip(), data.get("ctx_area_nk200", "").strip())
            if cursor == ("", "") or cursor in seen:
                raise RuntimeError("KIS_PAGINATION_INCOMPLETE")
            seen.add(cursor)
        raise RuntimeError("KIS_PAGINATION_LIMIT")

    def snapshot(self, instruments: dict) -> dict:
        holdings_raw = self.pages("/uapi/overseas-stock/v1/trading/inquire-balance", "TTTS3012R",
                                  {"OVRS_EXCG_CD": "NASD", "TR_CRCY_CD": "USD"}, "output1")
        pending_raw = self.pages("/uapi/overseas-stock/v1/trading/inquire-nccs", "TTTS3018R",
                                 {"OVRS_EXCG_CD": "NASD", "SORT_SQN": "DS"}, "output")
        holdings = []
        for row in holdings_raw:
            quantity = number(row.get("ovrs_cblc_qty"))
            if quantity > 0:
                if row.get("tr_crcy_cd") != "USD":
                    raise RuntimeError("NON_USD_HOLDING")
                holdings.append({"symbol": row["ovrs_pdno"], "quantity": str(quantity),
                                 "market_value_usd": str(number(row.get("ovrs_stck_evlu_amt")))})
        quotes = {}
        cash_limits = []
        for item in instruments.values():
            symbol, exchange = item["symbol"], item["exchange"]
            quote, _ = self.get("/uapi/overseas-price/v1/quotations/price", "HHDFS00000300",
                                {"AUTH": "", "EXCD": {"NASD": "NAS", "NYSE": "NYS", "AMEX": "AMS"}[exchange], "SYMB": symbol})
            price = number(quote.get("output", {}).get("last"))
            if price <= 0:
                raise RuntimeError("KIS_INVALID_QUOTE")
            # OVRS_ORD_UNPR는 최대 8자 필드다(실측: "1070.4500"=9자 거부, "999.4500"=8자 통과).
            # 시세 원문의 소수 자리수가 가격대에 따라 달라 그대로 넘기면 고가 종목에서
            # OPSQ2002(INPUT_FILED_SIZE)로 거부된다. 센트 단위로 정규화해 전달한다.
            cent_price = price.quantize(Decimal("0.01"))
            amount, _ = self.get("/uapi/overseas-stock/v1/trading/inquire-psamount", "TTTS3007R",
                                 {"CANO": self.account, "ACNT_PRDT_CD": self.product, "OVRS_EXCG_CD": exchange,
                                  "OVRS_ORD_UNPR": str(cent_price), "ITEM_CD": symbol})
            balance = amount.get("output", {})
            if balance.get("tr_crcy_cd") != "USD":
                raise RuntimeError("KIS_CASH_CURRENCY_UNKNOWN")
            # 환전 후 한도를 쓰지 않는다. 자동 환전·원화 증거금 확대는 별도 결정 사항이다.
            cash_limits.append(number(balance.get("ord_psbl_frcr_amt")))
            quotes[symbol] = {"price": str(price), "exchange": exchange, "freshness": "BROKER_LAST_PRICE_NOT_EXECUTION_VALIDATED"}
        return {
            "observed_at": datetime.now(timezone.utc).isoformat(), "source": "KIS_LIVE_READ_ONLY",
            "account_suffix": self.account[-4:], "account_fingerprint": self.fingerprint,
            "holdings": holdings, "pending_order_count": len(pending_raw),
            "available_usd": str(min(cash_limits)) if cash_limits else None, "quotes": quotes,
            "execution_quote_validated": False, "orders_sent": False,
        }

    def domestic_snapshot(self) -> dict:
        params = {"CANO": self.account, "ACNT_PRDT_CD": self.product, "AFHR_FLPR_YN": "N", "OFL_YN": "",
                  "INQR_DVSN": "02", "UNPR_DVSN": "01", "FUND_STTL_ICLD_YN": "N",
                  "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01", "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""}
        rows, seen = [], set()
        for page in range(20):
            data, continuation = self.get("/uapi/domestic-stock/v1/trading/inquire-balance", "TTTC8434R", params, "N" if page else "")
            if "output1" not in data or not data.get("output2"):
                raise RuntimeError("DOMESTIC_BALANCE_INCOMPLETE")
            for row in data["output1"]:
                qty = number(row.get("hldg_qty"))
                if qty > 0:
                    rows.append({"symbol": row["pdno"], "name": row["prdt_name"], "quantity": str(qty),
                                 "sellable_quantity": str(number(row.get("ord_psbl_qty"))),
                                 "last_price_krw": str(number(row.get("prpr"))),
                                 "value_krw": str(number(row.get("evlu_amt")))})
            summary = data["output2"][0]
            if continuation not in {"M", "F"}:
                return {"observed_at": datetime.now(timezone.utc).isoformat(), "account_suffix": self.account[-4:],
                        "source": "KIS_LIVE_READ_ONLY", "holdings": rows,
                        "cash_krw": str(number(summary.get("dnca_tot_amt"))),
                        "total_equity_krw": str(number(summary.get("tot_evlu_amt"))),
                        "orders_sent": False}
            cursor = (data.get("ctx_area_fk100", "").strip(), data.get("ctx_area_nk100", "").strip())
            if cursor == ("", "") or cursor in seen:
                raise RuntimeError("DOMESTIC_PAGINATION_INCOMPLETE")
            seen.add(cursor)
            params.update(CTX_AREA_FK100=cursor[0], CTX_AREA_NK100=cursor[1])
        raise RuntimeError("DOMESTIC_PAGINATION_LIMIT")

    def find_us_instrument(self, symbol: str, cusip: str) -> dict:
        if not symbol or len(cusip) != 9:
            raise ValueError("INVALID_US_INSTRUMENT_ID")
        # 해당 거래소에 상장되지 않은 종목을 조회하면 KIS는 rt_cd='7', msg_cd='APBN0024'
        # ("조회된 데이터가 없습니다")로 정상 응답한다. 이는 다른 두 거래소를 마저 확인해야
        # 하는 기대된 결과이며 실패가 아니다. 그 외 오류만 조회 불능으로 취급한다.
        NOT_LISTED_HERE = "KIS_READ_REJECTED:APBN0024"
        matches, errors = [], []
        for product_type, exchange in (("512", "NASD"), ("513", "NYSE"), ("529", "AMEX")):
            try:
                data, _ = self.get("/uapi/overseas-price/v1/quotations/search-info", "CTPF1702R",
                                   {"PRDT_TYPE_CD": product_type, "PDNO": symbol})
            except RuntimeError as exc:
                if str(exc) != NOT_LISTED_HERE:
                    errors.append(str(exc))
                continue
            row = data.get("output", {})
            standard = str(row.get("std_pdno", ""))
            if standard.startswith("US" + cusip):
                matches.append({"symbol": symbol, "cusip": cusip, "exchange": exchange,
                                "product_type": product_type, "standard_code": standard,
                                "name": row.get("prdt_name") or row.get("ovrs_item_name")})
        # 미상장 응답 외의 이유로 한 거래소라도 조회하지 못하면 다른 시장의 동명 종목을
        # 배제할 수 없다.
        if errors:
            raise RuntimeError("KIS_INSTRUMENT_LOOKUP_UNAVAILABLE")
        if len(matches) != 1:
            raise RuntimeError("US_INSTRUMENT_NOT_UNIQUELY_VERIFIED")
        return matches[0]

    def integrated_margin_snapshot(self) -> dict:
        data, _ = self.get("/uapi/domestic-stock/v1/trading/intgr-margin", "TTTC0869R", {
            "CANO": self.account, "ACNT_PRDT_CD": self.product, "CMA_EVLU_AMT_ICLD_YN": "N",
            "WCRC_FRCR_DVSN_CD": "01", "FWEX_CTRT_FRCR_DVSN_CD": "01",
        })
        row = data.get("output", {})
        return {
            "subscription": row.get("ovrs_stck_itgr_mgna_dvsn_name"),
            "usd_integrated_orderable": str(number(row.get("usd_itgr_ord_psbl_amt"))),
            "usd_reference_fx": str(number(row.get("usd_frst_bltn_exrt"))),
        }
