import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest

from guru_tracker.order_gateway import OrderGateway, payload


def fixtures(tmp_path, handler, enabled=True):
    broker = SimpleNamespace(account='00000000', product='01', key='TEST', secret='TEST',
                             fingerprint='TEST_ACCOUNT', token=lambda: 'MOCK_ONLY',
                             client=httpx.Client(transport=httpx.MockTransport(handler)))
    gateway = OrderGateway(broker, tmp_path/'ledger.sqlite', enabled=enabled)
    order = dict(intent_id='TEST-QUARTER-A', symbol='TEST', exchange='NASD',
                 side='BUY', quantity=1, limit_price_usd='10.00')
    now = datetime(2026, 9, 22, 14, tzinfo=timezone.utc)
    preflight = dict(observed_at=now.isoformat(), account_fingerprint='TEST_ACCOUNT',
                     instrument_verified=True, session_open_verified=True, quote_verified=True,
                     cash_and_position_verified=True, no_pending_orders=True, plan_reviewed=True,
                     order_sha256=hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest())
    return gateway, order, preflight, now


def test_disabled_never_posts(tmp_path):
    gateway, order, pre, now = fixtures(tmp_path, lambda _: pytest.fail('network'), False)
    with pytest.raises(ValueError, match='EXECUTION_DISABLED'):
        gateway.submit(order, preflight=pre, now=now)
    gateway.close()


def test_accept_is_not_fill_and_duplicate_not_posted(tmp_path):
    calls = []
    def handler(req):
        calls.append(req)
        assert req.headers['tr_id'] == 'TTTT1002U'
        assert json.loads(req.content)['ORD_DVSN'] == '00'
        return httpx.Response(200, json={'rt_cd': '0', 'output': {'ODNO': 'TEST_ORDER'}})
    gateway, order, pre, now = fixtures(tmp_path, handler)
    assert gateway.submit(order, preflight=pre, now=now)['state'] == 'ACCEPTED_NOT_FILLED'
    assert gateway.submit(order, preflight=pre, now=now)['request_sent'] is False
    assert len(calls) == 1
    gateway.close()


def test_timeout_persistent_no_retry(tmp_path):
    calls = []
    def handler(req):
        calls.append(req)
        raise httpx.ReadTimeout('test timeout')
    gateway, order, pre, now = fixtures(tmp_path, handler)
    assert gateway.submit(order, preflight=pre, now=now)['state'] == 'UNKNOWN_REQUIRES_RECONCILIATION'
    gateway.close()
    gateway, order, pre, now = fixtures(tmp_path, handler)
    assert gateway.submit(order, preflight=pre, now=now)['request_sent'] is False
    assert len(calls) == 1
    gateway.close()


@pytest.mark.parametrize('key,value', [('quote_verified', False), ('no_pending_orders', False),
                                      ('account_fingerprint', 'WRONG'), ('order_sha256', 'WRONG'),
                                      ('observed_at', '2026-09-22T13:00:00+00:00')])
def test_bad_preflight_never_posts(tmp_path, key, value):
    gateway, order, pre, now = fixtures(tmp_path, lambda _: pytest.fail('network'))
    pre[key] = value
    with pytest.raises(ValueError):
        gateway.submit(order, preflight=pre, now=now)
    gateway.close()


@pytest.mark.parametrize('qty', [0, -1, True, 1.5])
def test_bad_quantity(qty):
    broker = SimpleNamespace(account='TEST', product='01')
    with pytest.raises(ValueError):
        payload(broker, dict(symbol='TEST', exchange='NYSE', side='SELL', quantity=qty,
                             limit_price_usd='10.00'))


def test_second_intent_blocked_until_reconciled(tmp_path):
    calls = []
    def handler(req):
        calls.append(req)
        return httpx.Response(200, json={'rt_cd': '0', 'output': {'ODNO': 'TEST_ORDER'}})
    gateway, order, pre, now = fixtures(tmp_path, handler)
    gateway.submit(order, preflight=pre, now=now)
    order['intent_id'] = 'ANOTHER_ORDER'
    pre['order_sha256'] = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match='UNRESOLVED_ORDER_EXISTS'):
        gateway.submit(order, preflight=pre, now=now)
    assert len(calls) == 1
    gateway.close()


def test_intent_mutation_rejected(tmp_path):
    gateway, order, pre, now = fixtures(tmp_path, lambda _: httpx.Response(
        200, json={'rt_cd': '0', 'output': {'ODNO': 'TEST_ORDER'}}))
    gateway.submit(order, preflight=pre, now=now)
    order['quantity'] = 2
    pre['order_sha256'] = hashlib.sha256(json.dumps(order, sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError, match='INTENT_MUTATION_PROHIBITED'):
        gateway.submit(order, preflight=pre, now=now)
    gateway.close()
