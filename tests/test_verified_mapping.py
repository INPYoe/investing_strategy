import pytest
from guru_tracker.study import normalize_holdings


@pytest.mark.parametrize('old,cusip,name,kind,new', [
    ('0A2S.IL', '722304102', 'PDD HOLDINGS INC', 'SPONSORED ADS', 'PDD'),
    ('BN.TO', '11271J107', 'BROOKFIELD CORP', 'CL A LTD VT SH', 'BN'),
])
def test_verified_us_mapping_keeps_original(old, cusip, name, kind, new):
    frame = normalize_holdings([[old, name, kind, cusip, 100., 100., 10., None, None]])
    assert frame.iloc[0].symbol == new
    assert frame.iloc[0].provider_symbol == old
    assert frame.iloc[0].mapping_source.startswith('https://')


def test_mapping_rejects_wrong_security_class():
    with pytest.raises(ValueError, match='SECURITY_CLASS_MISMATCH'):
        normalize_holdings([['WRONG', 'PDD', 'ORDINARY', '722304102', 100., 100., 10., None, None]])
