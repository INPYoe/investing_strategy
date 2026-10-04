import numpy as np
import pandas as pd
import pytest

from guru_tracker.portfolio_study import eligible_holdings, rebalance, simulate_portfolio, target_weights


def holdings(weights):
    return pd.DataFrame([{"cusip": s, "symbol": s, "weight_pct": w} for s, w in weights.items()])


def test_exit_thresholds_and_no_new_small_positions():
    frame = holdings({"A": 4.9, "B": 3, "C": 2, "D": 6})
    assert set(eligible_holdings(frame, {"A", "B"}, 5).cusip) == {"D"}
    assert set(eligible_holdings(frame, {"A", "B"}, 3).cusip) == {"A", "B", "D"}
    assert set(eligible_holdings(frame, {"B", "C", "REMOVED"}, 0).cusip) == {"B", "C", "D"}


def test_equal_cap_and_cash_not_redistributed():
    assert target_weights(holdings({"A": 80, "B": 10}), "equal", .35) == {"A": .35, "B": .35}
    weights = target_weights(holdings({"A": 80, "B": 10}), "guru", .35)
    assert weights["A"] == .35
    assert weights["B"] == pytest.approx(.98 / 9)
    assert sum(target_weights(holdings({"A": 10, "B": 10, "C": 10}), "equal", .35).values()) == pytest.approx(.98)


def test_rebalance_cost_and_cash_accounting():
    targets, cash, fee, traded = rebalance({}, 10000, {"A": .5, "B": .48}, .001)
    assert sum(targets.values()) + cash + fee == pytest.approx(10000)
    assert fee == pytest.approx(traded * .001)
    assert targets["A"] / (10000 - fee) == pytest.approx(.5)
    targets, cash, fee, traded = rebalance({"A": 8000}, 2000, {"B": .98}, .001)
    assert "A" not in targets
    assert cash >= 0
    assert fee == pytest.approx((8000 + targets["B"]) * .001)


@pytest.mark.parametrize("weights,cost", [({"A": 1.1}, .001), ({"A": -1}, .001), ({"A": np.nan}, .001), ({}, -.1)])
def test_invalid_portfolio(weights, cost):
    with pytest.raises(ValueError):
        rebalance({}, 10000, weights, cost)


def fixture_prices():
    calendar = pd.bdate_range("2025-01-06", periods=6)
    prices = {s: pd.DataFrame({"adj_open": 100., "adj_close": 100.}, index=calendar) for s in ["A", "B"]}
    events = [{"accepted_at": "2025-01-06T14:00:00Z", "accession": "1", "holdings": holdings({"A": 8})},
              {"accepted_at": "2025-01-08T22:00:00Z", "accession": "2", "holdings": holdings({"B": 9})}]
    return events, prices, calendar


def test_no_lookahead_and_constant_capital():
    events, prices, calendar = fixture_prices()
    summary, curve, trades = simulate_portfolio(events, prices, calendar, allocation="equal", cap=1., exit_below=5, cost=0)
    assert trades.iloc[0].date == "2025-01-07"
    assert trades.loc[trades.accession.eq("2"), "date"].unique().tolist() == ["2025-01-09"]
    assert curve.nav.eq(10000).all()
    assert summary["total_return_pct"] == pytest.approx(0)
    assert summary["trade_count"] == 3
    assert (curve.cash >= 0).all()


def test_missing_price_fails_not_zero_filled():
    events, prices, calendar = fixture_prices()
    prices["A"].loc["2025-01-08", "adj_close"] = np.nan
    with pytest.raises(ValueError, match="INVALID_PRICE"):
        simulate_portfolio(events, prices, calendar, allocation="equal", cap=1., exit_below=5, cost=0)


def test_future_prices_do_not_change_prior_orders():
    events, prices, calendar = fixture_prices()
    kwargs = dict(allocation="equal", cap=1., exit_below=5, cost=.001)
    _, _, before = simulate_portfolio(events, prices, calendar, **kwargs)
    prices["A"].loc["2025-01-08":, "adj_close"] = 1000
    _, _, after = simulate_portfolio(events, prices, calendar, **kwargs)
    pd.testing.assert_frame_equal(before.loc[before.accession.eq("1")], after.loc[after.accession.eq("1")])


def test_terminal_cost_included():
    events, prices, calendar = fixture_prices()
    summary, curve, _ = simulate_portfolio(events[:1], prices, calendar, allocation="equal", cap=1., exit_below=5, cost=.001)
    invested = .98 * 10000 / (1 + .98 * .001)
    assert curve.nav.iloc[-1] == pytest.approx(10000 - invested * .002)
    assert summary["cost_usd"] == pytest.approx(invested * .002)


def test_drawdown_includes_initial_nav():
    events, prices, calendar = fixture_prices()
    prices["A"].loc[:, "adj_close"] = 50
    summary, _, _ = simulate_portfolio(events[:1], prices, calendar, allocation="equal", cap=1., exit_below=5, cost=0)
    assert summary["max_drawdown_pct"] == pytest.approx(-49)


def test_post_disclosure_holdings_exit_is_not_future_leakage():
    events, prices, calendar = fixture_prices()
    events[1]["holdings"] = holdings({"A": 4, "B": 9})
    _, _, trades5 = simulate_portfolio(events, prices, calendar, allocation="equal", cap=1., exit_below=5, cost=0)
    _, _, trades0 = simulate_portfolio(events, prices, calendar, allocation="equal", cap=1., exit_below=0, cost=0)
    assert trades5.loc[trades5.symbol.eq("A")].iloc[-1].target_weight == 0
    assert trades0.loc[trades0.symbol.eq("A")].iloc[-1].target_weight == pytest.approx(.49)
