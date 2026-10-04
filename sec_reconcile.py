"""최신 보유표를 SEC XML 원문과 대조한다. 실패를 공개 fallback 검증으로 대체하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from guru_tracker.study import Fetcher, ROOT


def parse_information_table(text: str) -> pd.DataFrame:
    root = ET.fromstring(text)
    rows = []
    for element in root.iter():
        if element.tag.rsplit('}', 1)[-1] != 'infoTable':
            continue
        fields = {node.tag.rsplit('}', 1)[-1]: (node.text or '').strip() for node in element.iter()}
        if fields.get('putCall') or fields.get('sshPrnamtType') != 'SH':
            continue
        rows.append({'cusip': fields['cusip'], 'shares': float(fields['sshPrnamt']),
                     'value_usd': float(fields['value'])})
    if not rows:
        raise ValueError('NO_SPOT_INFORMATION_TABLE')
    frame = pd.DataFrame(rows).groupby('cusip', as_index=False)[['shares', 'value_usd']].sum()
    values = frame[['shares', 'value_usd']].to_numpy()
    if not np.isfinite(values).all() or (frame.shares <= 0).any() or (frame.value_usd < 0).any() or frame.value_usd.sum() <= 0:
        raise ValueError('INVALID_OFFICIAL_VALUES')
    frame['weight_pct'] = frame.value_usd / frame.value_usd.sum() * 100
    return frame


def compare(official, provider):
    if official.cusip.duplicated().any() or provider.cusip.duplicated().any():
        raise ValueError('DUPLICATE_CUSIP')
    joined = official.merge(provider, on='cusip', how='outer', suffixes=('_sec', '_provider'), indicator=True)
    missing = joined['_merge'].ne('both')
    shares_ok = np.isclose(joined.shares_sec, joined.shares_provider, rtol=0, atol=1e-6)
    weights_ok = np.isclose(joined.weight_pct_sec, joined.weight_pct_provider, rtol=0, atol=.01)
    joined['shares_match'] = shares_ok & ~missing
    joined['weights_match'] = weights_ok & ~missing
    joined['sec_usd_per_provider_value_unit'] = joined.value_usd / joined.provider_value_thousands_usd.replace(0, np.nan)
    joined['status'] = np.where(joined.shares_match & joined.weights_match, 'MATCH_SHARES_WEIGHTS', 'MISMATCH')
    return joined


def text_document(fetch, url):
    path = fetch.root/'cache'/(hashlib.sha256(url.encode()).hexdigest()+'.json')
    if path.exists():
        doc = json.loads(path.read_text())
        if doc.get('url') != url:
            raise ValueError('CACHE_URL_MISMATCH')
        return doc['data']['text']
    for attempt in range(4):
        try:
            time.sleep(.4)
            response = fetch.client.get(url)
            response.raise_for_status()
            text = response.text
            ET.fromstring(text)
            path.write_text(json.dumps({'url': url, 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                                        'data': {'text': text}}))
            fetch.attempts.append({'url': url, 'status': 'OK', 'retry_count': attempt})
            return text
        except httpx.HTTPError as exc:
            code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            fetch.attempts.append({'url': url, 'status': 'FAILED', 'http_status': code, 'retry_count': attempt})
            if code in {400, 401, 403, 404} or attempt == 3:
                raise
            time.sleep((2, 5, 15)[attempt])


def run(report):
    holdings = pd.read_csv(report/'current_holdings.csv', dtype={'cusip': str, 'accession': str})
    registry = json.loads((ROOT/'managers.json').read_text())
    fetch = Fetcher(ROOT/'data')
    statuses, results = [], []
    try:
        for manager in registry['production_research_cohort']:
            selected = holdings.loc[holdings.manager_id.eq(manager['id'])]
            try:
                if selected.empty or selected.accession.nunique() != 1:
                    raise ValueError('ONE_ACCESSION_REQUIRED')
                if (pd.to_datetime(selected.report_date) < pd.Timestamp('2023-03-31')).any():
                    raise ValueError('LEGACY_VALUE_UNITS_REQUIRE_EXPLICIT_SCHEMA')
                accession = selected.accession.iloc[0].replace('-', '')
                base = f'https://www.sec.gov/Archives/edgar/data/{int(manager["cik"])}/{accession}/'
                listing = fetch.get(base+'index.json')
                names = [item['name'] for item in listing['directory']['item']
                         if item['name'].endswith('.xml') and '/' not in item['name']]
                tables = []
                for name in names:
                    try:
                        frame = parse_information_table(text_document(fetch, base+name))
                        tables.append((frame, base+name))
                    except ValueError as exc:
                        if str(exc) != 'NO_SPOT_INFORMATION_TABLE':
                            raise
                if len(tables) != 1:
                    raise ValueError('UNIQUE_INFORMATION_TABLE_REQUIRED')
                frame, url = tables[0]
                reconciled = compare(frame, selected).assign(manager=manager['id'], official_url=url)
                results.append(reconciled)
                status = 'MATCH_SHARES_WEIGHTS_VALUE_UNIT_REVIEW' if reconciled.status.eq('MATCH_SHARES_WEIGHTS').all() else 'MISMATCH'
                statuses.append({'manager': manager['id'], 'status': status, 'official_url': url,
                                 'rows': len(reconciled), 'error': ''})
            except Exception as exc:
                statuses.append({'manager': manager['id'], 'status': 'UNVERIFIED', 'error': type(exc).__name__})
    finally:
        out = report.parent/'validation'
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(statuses).to_csv(out/'sec_reconciliation_status.csv', index=False)
        if results:
            pd.concat(results).to_csv(out/'sec_reconciliation_rows.csv', index=False)
        (out/'sec_source_attempts.json').write_text(json.dumps(fetch.attempts, ensure_ascii=False, indent=2))
        fetch.client.close()
    return statuses


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser(description='최신 SEC 보유 원문 대조, 주문 없음')
    parser.add_argument('--report', type=Path, required=True)
    print(json.dumps(run(parser.parse_args().report), ensure_ascii=False, indent=2))
