import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_investment_targets_and_safety_constraints():
    policy = json.loads((ROOT / "multi_guru_policy.json").read_text())
    deployment = policy["capital_deployment"]
    assert sum(deployment["phase_1_targets_pct"].values()) == 100
    assert sum(deployment["phase_2_targets_pct"].values()) == 100
    assert deployment["cash_target_pct"] == 0
    assert deployment["minimum_invested_pct"] == 98
    assert deployment["allow_margin_borrowing"] is False
    assert deployment["allow_market_orders"] is False
    assert deployment["phase_1_targets_pct"] == {"SPCX": 40, "INTC": 31, "CBRS": 29}


def test_consensus_names_enter_only_after_confirmation():
    policy = json.loads((ROOT / "multi_guru_policy.json").read_text())
    deployment = policy["capital_deployment"]
    assert not {"TSM", "AMZN", "META"} & deployment["phase_1_targets_pct"].keys()
    assert {"TSM", "AMZN", "META"} <= deployment["phase_2_targets_pct"].keys()
    assert "CONFIRMED" in deployment["phase_2_gate"]
