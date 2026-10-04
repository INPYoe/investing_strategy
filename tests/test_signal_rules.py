import pytest

from guru_tracker.signal_rules import loss_alert, position_signal


def signal(previous=None, current=None, **kwargs):
    defaults = dict(manager_id="manager-a", accession="filing-a", cusip="CUSIP-A",
                    previous=previous, current=current, comparison_verified=True, split_factor=1)
    defaults.update(kwargs)
    return position_signal(**defaults)


def row(weight, shares=100):
    return {"weight_pct": weight, "shares": shares}


def test_new_entry_threshold_three_percent_inclusive():
    assert signal(current=row(3))["signal"] == "NEW_ENTRY_CANDIDATE"
    assert signal(current=row(2.99))["signal"] == "NO_ACTION"


def test_doubling_requires_both_and_no_new_position_floor():
    assert signal(row(1, 100), row(2, 200))["signal"] == "INCREASE_ENTRY_CANDIDATE"
    assert signal(row(3, 100), row(6, 100))["signal"] == "REVIEW"
    assert signal(row(3, 100), row(3, 200))["signal"] == "REVIEW"


def test_half_reduction_inclusive_and_not_percentage_points():
    result = signal(row(8, 100), row(4, 50))
    assert result["signal"] == "REDUCE_HALF"
    assert result["reduction_fraction"] == .5
    assert result["position_scope"] == "MANAGER_ATTRIBUTED_POSITION_ONLY"
    assert signal(row(8, 100), row(4, 60))["signal"] == "REVIEW"


def test_below_old_five_percent_is_not_exit():
    assert signal(row(5.1), row(4.9))["signal"] == "NO_ACTION"


def test_split_is_not_buy_and_reverse_split_is_not_sale():
    assert signal(row(5, 100), row(5, 200), split_factor=2)["signal"] == "NO_ACTION"
    assert signal(row(5, 100), row(5, 10), split_factor=.1)["signal"] == "NO_ACTION"
    assert signal(row(5), row(10, 200), split_factor=None)["signal"] == "REVIEW"


def test_exit_requires_more_than_missing_row():
    assert signal(row(5), None)["signal"] == "REVIEW"
    result = signal(row(5), None, absence_reviewed=True)
    assert result["signal"] == "EXIT_DISCLOSED_POSITION"
    assert result["reduction_fraction"] == 1
    assert not result["orders_sent"]


def test_incomplete_comparison_never_generates_action():
    assert signal(row(5), None, absence_reviewed=True, comparison_verified=False)["signal"] == "REVIEW"
    assert signal(current=row(5), comparison_verified=False)["signal"] == "REVIEW"


def test_manager_identity_prevents_cross_manager_deduplication():
    a = signal(current=row(5))
    b = signal(current=row(5), manager_id="manager-b")
    assert a["id"] != b["id"]
    assert a["id"] == signal(current=row(5))["id"]


@pytest.mark.parametrize("current", [row(float("nan")), row(101), row(5, 0)])
def test_invalid_input_not_signal(current):
    with pytest.raises(ValueError):
        signal(current=current)


def test_twenty_percent_is_alert_only_including_gap():
    assert loss_alert(100, 80.01, same_currency=True, price_verified=True)["status"] == "NO_ALERT"
    for price in (80, 60):
        result = loss_alert(100, price, same_currency=True, price_verified=True)
        assert result["status"] == "ALERT_AND_REVIEW"
        assert not result["automatic_sell"]
        assert not result["orders_sent"]


def test_invalid_quote_does_not_issue_loss_alert():
    assert loss_alert(100, 70, same_currency=False, price_verified=True)["status"] == "DATA_REVIEW"
    assert loss_alert(100, 70, same_currency=True, price_verified=False)["status"] == "DATA_REVIEW"
