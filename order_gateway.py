"""사용자 운영 프로그램용 미국 지정가 주문 어댑터. CLI/스케줄러와 연결하지 않는다.

실계좌 활성화 기본값은 False다. 호출자는 체결대사, 호가/세션/잔고 검증을 수행해야 한다.
테스트는 MockTransport만 사용하며 실제 증권사로 요청하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from guru_tracker.kis_readonly import BASE, number


def payload(broker, order: dict) -> tuple[str, dict]:
    if order.get('side') not in {'BUY', 'SELL'}:
        raise ValueError('INVALID_SIDE')
    if order.get('exchange') not in {'NASD', 'NYSE', 'AMEX'}:
        raise ValueError('US_EXCHANGE_REQUIRED')
    if not re.fullmatch(r'[A-Z][A-Z0-9.\-]{0,14}', order.get('symbol', '')):
        raise ValueError('INVALID_SYMBOL')
    qty = order.get('quantity')
    if type(qty) is not int or qty <= 0:
        raise ValueError('POSITIVE_INTEGER_QUANTITY_REQUIRED')
    price = number(order.get('limit_price_usd'))
    if price < 1 or price != price.quantize(Decimal('.01')):
        raise ValueError('ONLY_CENT_LIMIT_ORDERS_AT_LEAST_ONE_DOLLAR_SUPPORTED')
    return ('TTTT1002U' if order['side'] == 'BUY' else 'TTTT1006U'), {
        'CANO': broker.account, 'ACNT_PRDT_CD': broker.product,
        'OVRS_EXCG_CD': order['exchange'], 'PDNO': order['symbol'],
        'ORD_QTY': str(qty), 'OVRS_ORD_UNPR': str(price),
        'ORD_DVSN': '00', 'ORD_SVR_DVSN_CD': '0',
        'SLL_TYPE': '' if order['side'] == 'BUY' else '00',
        'CTAC_TLNO': '', 'MGCO_APTM_ODNO': '',
    }


class OrderGateway:
    def __init__(self, broker, ledger: Path, *, enabled: bool = False):
        self.broker, self.enabled = broker, enabled
        ledger.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(ledger, timeout=20)
        if self.db.execute('PRAGMA journal_mode=WAL').fetchone()[0] != 'wal':
            raise RuntimeError('WAL_REQUIRED')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS submissions '
                        '(id TEXT PRIMARY KEY, fingerprint TEXT, state TEXT, order_no TEXT)')
        self.db.commit()

    def close(self):
        self.db.close()

    def submit(self, order: dict, *, preflight: dict, now: datetime | None = None) -> dict:
        if self.enabled is not True:
            raise ValueError('EXECUTION_DISABLED')
        now = now or datetime.now(timezone.utc)
        stamp = datetime.fromisoformat(preflight['observed_at'])
        if stamp.tzinfo is None or not 0 <= (now - stamp).total_seconds() <= 30:
            raise ValueError('FRESH_PREFLIGHT_REQUIRED')
        if preflight.get('account_fingerprint') != self.broker.fingerprint:
            raise ValueError('ACCOUNT_MISMATCH')
        required = ('instrument_verified', 'session_open_verified', 'quote_verified',
                    'cash_and_position_verified', 'no_pending_orders', 'plan_reviewed')
        if any(preflight.get(key) is not True for key in required):
            raise ValueError('PREFLIGHT_INCOMPLETE')
        tr_id, body = payload(self.broker, order)
        digest = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
        if preflight.get('order_sha256') != digest:
            raise ValueError('PREFLIGHT_ORDER_MISMATCH')
        if not order.get('intent_id') or not isinstance(order['intent_id'], str):
            raise ValueError('STABLE_INTENT_ID_REQUIRED')
        identity = self.broker.fingerprint + ':' + order['intent_id']
        # 응답이 유실되어도 재전송하지 않는다. 재시작 후에는 체결 대사가 필요하다.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            old = self.db.execute('SELECT fingerprint, state, order_no FROM submissions WHERE id=?',
                                  (identity,)).fetchone()
            if old:
                if old[0] != digest:
                    raise ValueError('INTENT_MUTATION_PROHIBITED')
                return {'state': old[1], 'order_no': old[2], 'request_sent': False}
            if self.db.execute('SELECT 1 FROM submissions WHERE state IN (?, ?, ?)',
                               ('SUBMITTING', 'UNKNOWN_REQUIRES_RECONCILIATION',
                                'ACCEPTED_NOT_FILLED')).fetchone():
                raise ValueError('UNRESOLVED_ORDER_EXISTS')
            self.db.execute('INSERT INTO submissions VALUES (?, ?, ?, ?)',
                            (identity, digest, 'SUBMITTING', ''))
        state, order_no = 'UNKNOWN_REQUIRES_RECONCILIATION', ''
        try:
            response = self.broker.client.post(BASE + '/uapi/overseas-stock/v1/trading/order',
                json=body, headers={'authorization': 'Bearer ' + self.broker.token(),
                    'appkey': self.broker.key, 'appsecret': self.broker.secret,
                    'tr_id': tr_id, 'custtype': 'P', 'content-type': 'application/json'})
            if response.status_code == 200:
                data = response.json()
                if data.get('rt_cd') == '0':
                    output = data.get('output', {})
                    order_no = str(output.get('ODNO') or output.get('odno') or '')
                    if order_no:
                        state = 'ACCEPTED_NOT_FILLED'
                elif data.get('rt_cd') == '1':
                    state = 'REJECTED'
        except Exception:
            # 주문 POST에는 자동 재시도를 적용하지 않는다.
            pass
        with self.db:
            self.db.execute('UPDATE submissions SET state=?, order_no=? WHERE id=?',
                            (state, order_no, identity))
        return {'state': state, 'order_no': order_no, 'request_sent': True}
