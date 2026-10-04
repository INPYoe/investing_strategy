"""승인된 기존 한국투자 전략 예약 작업 중단. plist 파일은 삭제하지 않는다."""
import json
import os
import subprocess
from datetime import datetime, timezone

from guru_tracker.study import ROOT

LABELS = [
    "com.krx.canslim.daily", "com.krx.canslim.monitor",
    "com.krx.canslim.stop_placer", "com.krx.canslim.weekly_exec",
    "com.investingresearcher.krx-vcp-pretrade", "com.investingresearcher.krx-vcp-poll",
    "com.investingresearcher.krx-vcp-scan",
]


def main():
    uid = os.getuid()
    records = []
    for label in LABELS:
        service = f"gui/{uid}/{label}"
        disabled = subprocess.run(["launchctl", "disable", service], capture_output=True, text=True)
        unloaded = subprocess.run(["launchctl", "bootout", service], capture_output=True, text=True)
        probe = subprocess.run(["launchctl", "print", service], capture_output=True, text=True)
        record = {"label": label, "disable_returncode": disabled.returncode,
                  "bootout_returncode": unloaded.returncode, "loaded_after": probe.returncode == 0}
        records.append(record)
    state = subprocess.run(["launchctl", "print-disabled", f"gui/{uid}"], capture_output=True, text=True)
    for record in records:
        record["disabled_confirmed"] = f'"{record["label"]}" => enabled' not in state.stdout and f'"{record["label"]}" => disabled' in state.stdout
        # macOS 버전에 따라 boolean으로 출력한다.
        record["disabled_confirmed"] |= f'"{record["label"]}" => true' in state.stdout
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "user_authorized": True,
               "action": "STOP_LEGACY_STRATEGY", "services": records,
               "broker_orders_cancelled": False, "positions_sold": False}
    path = ROOT / "data" / "automation" / "legacy_stop.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if any(r["disable_returncode"] or r["loaded_after"] or not r["disabled_confirmed"] for r in records):
        raise SystemExit("LEGACY_STOP_VERIFICATION_FAILED")


if __name__ == "__main__":
    main()
