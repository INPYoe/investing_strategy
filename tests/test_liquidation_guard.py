from datetime import datetime
from zoneinfo import ZoneInfo

from guru_tracker.liquidation import tick


def test_disabled_configuration_blocks_live_execution_before_broker_calls():
    result = tick({"enabled": False}, object(), object(), execute=True,
                  now=datetime(2026, 9, 22, 10, tzinfo=ZoneInfo("Asia/Seoul")))
    assert result == {"status": "DISABLED", "orders_sent": False}
