"""분기별 연구 실행기. 증권사 접속이나 주문 제출 기능은 없다."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from guru_tracker.multi_guru import expected_report_date, run as collect
from guru_tracker.selection import run as select
from guru_tracker.study import ROOT


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def scheduled_window(day: date) -> bool:
    return day.month in {2, 5, 8, 11} and 20 <= day.day <= 28


def run(day: date, *, state_dir: Path, manual: bool = False, collector=collect, selector=select) -> dict:
    if not manual and not scheduled_window(day):
        return {'status': 'OUTSIDE_QUARTER_WINDOW', 'orders_sent': False}
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / 'runner.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'ALREADY_RUNNING', 'orders_sent': False}
        period = expected_report_date(day.isoformat())
        path = state_dir / (period + '.json')
        old = json.loads(path.read_text()) if path.exists() else {}
        if old.get('status') == 'RESEARCH_COMPLETE_REVIEW_REQUIRED':
            return {**old, 'status': 'ALREADY_RESEARCHED', 'orders_sent': False}
        result = {'period': period, 'asof': day.isoformat(),
                  'run_timestamp': datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                  'orders_sent': False, 'execution_enabled': False}
        try:
            report = collector(day.isoformat())
            meta = json.loads((report / 'metadata.json').read_text())
            if meta.get('sample_data_used') is not False:
                raise ValueError('SAMPLE_OR_UNKNOWN_DATA_PROHIBITED')
            requested = meta.get('managers_requested', 0)
            complete = (meta.get('status') == 'COMPLETE' and requested > 0
                        and meta.get('managers_collected') == requested
                        and meta.get('managers_unavailable') == 0)
            if complete:
                selection_output = selector(day.isoformat())
                sel_meta = json.loads((selection_output / 'metadata.json').read_text())
                result.update(report=str(report), selection=str(selection_output),
                              managers_requested=requested,
                              managers_collected=meta.get('managers_collected', 0),
                              eligible_candidates=sel_meta.get('eligible_candidates', 0),
                              selected_count=sel_meta.get('selected_count', 0),
                              status='RESEARCH_COMPLETE_REVIEW_REQUIRED',
                              next_action='기업행위·발행사 통합·배분 검증 후 주문 초안을 별도로 검토')
            else:
                result.update(report=str(report), managers_requested=requested,
                              managers_collected=meta.get('managers_collected', 0),
                              status='PARTIAL_RETRY_REQUIRED',
                              next_action='기업행위·발행사 통합·배분 검증 후 주문 초안을 별도로 검토')
        except Exception as exc:
            # 외부 예외 문자열에는 URL/키가 포함될 수 있으므로 저장하지 않는다.
            result.update(status='DATA_COLLECTION_FAILED', error_type=type(exc).__name__)
        atomic_json(path, result)
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description='분기 공시 연구. 주문 없음.')
    parser.add_argument('--asof', type=date.fromisoformat,
                        default=datetime.now(ZoneInfo('Asia/Seoul')).date())
    parser.add_argument('--manual', action='store_true', help='분기 실행창 밖에서 연구 실행')
    args = parser.parse_args()
    result = run(args.asof, state_dir=ROOT / 'data' / 'quarterly', manual=args.manual)
    print(json.dumps(result, ensure_ascii=False))
    return 2 if result['status'] in {'DATA_COLLECTION_FAILED', 'PARTIAL_RETRY_REQUIRED'} else 0


if __name__ == '__main__':
    raise SystemExit(main())
