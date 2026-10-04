import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("study", Path(__file__).parents[1] / "study.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def price_fixture():
    dates = pd.bdate_range("2024-01-02", periods=80)
    frame = pd.DataFrame({"adj_open": 100., "adj_close": 100.}, index=dates)
    return frame, dates


def test_weight_denominator_excludes_options_and_principal():
    raw = [
        ["A", "A", "COM", "1", 50, 0, 10, None, None],
        ["A", "A", "COM", "1", 50, 0, 10, None, None],
        ["B", "B", "COM", "2", 900, 0, 90, None, None],
        ["A", "A", "COM", "1", 2000, 0, 100, None, "call"],
        ["C", "C", "BOND", "3", 3000, 0, None, 100, None],
    ]
    result = study.normalize_holdings(raw)
    assert len(result) == 2
    assert result.set_index("symbol").loc["A", "weight_pct"] == 10
    assert result.weight_pct.sum() == 100


def test_entry_uses_us_date_and_skips_weekend():
    calendar = pd.DatetimeIndex(["2026-08-14", "2026-08-17", "2026-08-18"])
    assert study.first_session(calendar, "2026-08-14T19:55:57Z") == 1
    # UTC 이미 토요일이지만 뉴욕에서는 금요일이다.
    assert study.first_session(calendar, "2026-08-15T00:30:00Z") == 1


def test_known_wrong_provider_ticker_is_replaced_by_cusip():
    raw = [["REPYY", "YPF", "ADS", "984245100", 100, 100, 50, None, None]]
    row = study.normalize_holdings(raw).iloc[0]
    assert row.symbol == "YPF"
    assert row.provider_symbol == "REPYY"


def test_non_usd_price_rejected():
    class Fetch:
        def get(self, url):
            return {"chart": {"result": [{"meta": {"currency": "CAD"}}]}}
    with pytest.raises(ValueError, match="NON_USD"):
        study.yahoo_prices(Fetch(), "TECK-B.TO", "2021-01-01", "2026-09-21")


def test_cash_and_cost_accounting():
    frame, dates = price_fixture()
    expected = ((1 - .001) / (1 + .001) - 1) * 100
    for strategy in ["next_open", "wait_20", "split_0_5_20"]:
        result = study.simulate(frame, dates, 0, 63, strategy, .001)
        assert result["return_pct"] == pytest.approx(expected)
        assert result["exit_date"] == str(dates[63].date())
    cash = study.simulate(frame, dates, 0, 63, "dip_5pct_within_20", .001)
    assert not cash["filled"]
    assert cash["return_pct"] == 0


def test_dip_uses_next_open_not_signal_close_or_limit_price():
    frame, dates = price_fixture()
    frame.loc[dates[5], "adj_close"] = 94
    frame.loc[dates[6], "adj_open"] = 97
    result = study.simulate(frame, dates, 0, 63, "dip_5pct_within_20", 0)
    assert result["first_entry_date"] == str(dates[6].date())
    assert result["return_pct"] == pytest.approx((100 / 97 - 1) * 100)


def test_missing_prices_and_immature_events_not_imputed():
    frame, dates = price_fixture()
    with pytest.raises(ValueError, match="IMMATURE"):
        study.simulate(frame, dates, 0, 126, "next_open", 0)
    frame.loc[dates[4], "adj_close"] = np.nan
    with pytest.raises(ValueError, match="MISSING_PRICE"):
        study.simulate(frame, dates, 0, 63, "wait_20", 0)


def test_delayed_entry_common_exit_counts_missed_gains():
    frame, dates = price_fixture()
    frame.loc[dates[1]:, ["adj_open", "adj_close"]] = 120
    immediate = study.simulate(frame, dates, 0, 63, "next_open", 0)
    delayed = study.simulate(frame, dates, 0, 63, "wait_5", 0)
    assert immediate["return_pct"] == pytest.approx(20)
    assert delayed["return_pct"] == 0
    assert immediate["exit_date"] == delayed["exit_date"]
