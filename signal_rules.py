"""구루별 공시 신호 판정. 계좌 주문·전체 계좌 매도와 분리된 순수 함수다."""
from __future__ import annotations

import hashlib
from math import isfinite


def finite(value, name: str) -> float:
    number = float(value)
    if not isfinite(number) or number < 0:
        raise ValueError("INVALID_" + name)
    return number


def position_signal(*, manager_id: str, accession: str, cusip: str,
                    previous: dict | None, current: dict | None,
                    comparison_verified: bool, split_factor: float | None = None,
                    absence_reviewed: bool = False, minimum_new_weight: float = 3.) -> dict:
    """split_factor는 전분기 수량을 이번 분기말 주식 단위로 바꾸는 배수다.

    2:1 분할이면 2, 1:10 병합이면 0.1. 분할 없음도 검증된 1을 넘겨야 한다.
    missing row는 전량 매도 증거가 아니므로 별도의 부재 원인 검토가 필요하다.
    """
    if not manager_id or not accession or not cusip:
        raise ValueError("MISSING_SIGNAL_IDENTITY")
    minimum = finite(minimum_new_weight, "MINIMUM_WEIGHT")
    if not 0 < minimum <= 100:
        raise ValueError("INVALID_MINIMUM_WEIGHT")
    result = {"id": hashlib.sha256(f"{manager_id}:{accession}:{cusip}".encode()).hexdigest(),
              "manager_id": manager_id, "accession": accession, "cusip": cusip,
              "signal": "NO_ACTION", "evidence": [], "reduction_fraction": 0.,
              "position_scope": "MANAGER_ATTRIBUTED_POSITION_ONLY", "orders_sent": False}
    for label, row in (("PREVIOUS", previous), ("CURRENT", current)):
        if row is not None:
            weight, shares = finite(row["weight_pct"], label + "_WEIGHT"), finite(row["shares"], label + "_SHARES")
            if not 0 < weight <= 100 or shares <= 0:
                raise ValueError("INVALID_" + label + "_HOLDING")
    if not comparison_verified:
        return {**result, "signal": "REVIEW", "evidence": ["filing_or_identifier_comparison_unverified"]}
    if previous is None and current is None:
        return result
    if previous is not None and current is None:
        if not absence_reviewed:
            return {**result, "signal": "REVIEW", "evidence": ["missing_row_not_proof_of_exit"]}
        return {**result, "signal": "EXIT_DISCLOSED_POSITION", "reduction_fraction": 1.,
                "evidence": ["absent_current_public_holdings", "amendment_confidentiality_and_identifier_review_completed"]}
    if previous is None:
        if float(current["weight_pct"]) >= minimum:
            return {**result, "signal": "NEW_ENTRY_CANDIDATE",
                    "evidence": ["absent_previous_public_holdings", "new_weight_at_or_above_threshold"]}
        return {**result, "evidence": ["new_weight_below_threshold"]}
    if split_factor is None:
        return {**result, "signal": "REVIEW", "evidence": ["split_adjustment_unverified"]}
    split = finite(split_factor, "SPLIT_FACTOR")
    if split <= 0:
        raise ValueError("INVALID_SPLIT_FACTOR")
    weight_ratio = float(current["weight_pct"]) / float(previous["weight_pct"])
    shares_ratio = float(current["shares"]) / (float(previous["shares"]) * split)
    result.update(weight_ratio=weight_ratio, split_adjusted_shares_ratio=shares_ratio, split_factor=split)
    if weight_ratio <= .5 and shares_ratio <= .5:
        return {**result, "signal": "REDUCE_HALF", "reduction_fraction": .5,
                "evidence": ["weight_reduced_at_least_half", "adjusted_shares_reduced_at_least_half"]}
    if weight_ratio >= 2 and shares_ratio >= 2:
        return {**result, "signal": "INCREASE_ENTRY_CANDIDATE",
                "evidence": ["weight_at_least_doubled", "adjusted_shares_at_least_doubled"]}
    if weight_ratio <= .5 or shares_ratio <= .5 or weight_ratio >= 2 or shares_ratio >= 2:
        return {**result, "signal": "REVIEW", "evidence": ["weight_and_share_thresholds_disagree"]}
    return {**result, "evidence": ["no_confirmed_entry_or_reduction", "below_5pct_is_not_an_exit_rule"]}


def loss_alert(average_cost: float, current_price: float, *, same_currency: bool,
               price_verified: bool, threshold_pct: float = 20.) -> dict:
    cost, price = finite(average_cost, "AVERAGE_COST"), finite(current_price, "PRICE")
    threshold = finite(threshold_pct, "LOSS_THRESHOLD")
    if cost <= 0 or price <= 0 or not 0 < threshold < 100:
        raise ValueError("INVALID_LOSS_INPUT")
    if not same_currency or not price_verified:
        return {"status": "DATA_REVIEW", "automatic_sell": False, "orders_sent": False}
    loss = (price / cost - 1) * 100
    triggered = price <= cost * (1 - threshold / 100)
    return {"status": "ALERT_AND_REVIEW" if triggered else "NO_ALERT", "pnl_pct": loss,
            "reference": "AVERAGE_ENTRY_COST", "automatic_sell": False, "orders_sent": False}
