import pandas as pd
import pytest

from guru_tracker.sec_reconcile import parse_information_table, compare


def test_xml_filters_options_and_aggregates():
    text = '''<informationTable xmlns="urn:test">
    <infoTable><cusip>001</cusip><value>100</value><sshPrnamt>2</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
    <infoTable><cusip>001</cusip><value>100</value><sshPrnamt>2</sshPrnamt><sshPrnamtType>SH</sshPrnamtType></infoTable>
    <infoTable><cusip>002</cusip><value>500</value><sshPrnamt>5</sshPrnamt><sshPrnamtType>SH</sshPrnamtType><putCall>PUT</putCall></infoTable>
    </informationTable>'''
    frame = parse_information_table(text)
    assert len(frame) == 1
    assert frame.iloc[0].shares == 4
    assert frame.iloc[0].value_usd == 200
    assert frame.iloc[0].weight_pct == 100


def test_reconciliation_detects_units_and_missing_rows():
    official = pd.DataFrame({'cusip': ['001', '002'], 'shares': [2., 3.],
                             'value_usd': [100., 100.], 'weight_pct': [50., 50.]})
    provider = pd.DataFrame({'cusip': ['001'], 'shares': [2.],
                             'provider_value_thousands_usd': [.1], 'weight_pct': [100.]})
    result = compare(official, provider)
    assert result.iloc[0].sec_usd_per_provider_value_unit == pytest.approx(1000)
    assert result.status.eq('MISMATCH').all()
