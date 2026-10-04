#!/bin/sh
# cron에서 호출하는 연구 전용 진입점. 주문 모듈은 호출하지 않는다.
set -eu
ROOT='/Volumes/Extreme SSD/Investing_Researcher'
if [ ! -d "$ROOT/guru_tracker" ] || [ ! -x "$ROOT/.venv/bin/python" ]; then
    /usr/bin/logger -t guru-quarterly 'SSD or Python unavailable; nothing executed'
    exit 2
fi
cd "$ROOT"
mkdir -p guru_tracker/logs
exec "$ROOT/.venv/bin/python" -m guru_tracker.quarterly >> guru_tracker/logs/quarterly.log 2>&1
