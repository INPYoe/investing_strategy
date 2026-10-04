from guru_tracker.email_report import digest_context, render_html


def context():
    return {"report_date": "2026-09-21", "managers_collected": 4, "managers_requested": 5,
            "liquidation_status": "DOMESTIC_LIQUIDATION_COMPLETE", "domestic_holdings": [],
            "domestic_pending_order_count": 0, "usd_integrated_orderable": "7259.73",
            "targets": {"SPCX": 30.49, "TSM": 21.95},
            "directions": [{"symbol": "TSM", "candidate_level": "CORE",
                            "meaningful_manager_count": 4, "positive_signal_count": 2,
                            "reducing_manager_count": 0, "direction_state": "BROAD_ACCUMULATION"}],
            "orders_sent": False, "attachments": []}


def test_email_contains_account_allocation_and_direction():
    body = render_html(context())
    assert "DOMESTIC_LIQUIDATION_COMPLETE" in body
    assert "$7259.73" in body
    assert "SPCX" in body and "TSM" in body
    assert "BROAD_ACCUMULATION" in body


def test_email_digest_ignores_attachment_paths():
    left, right = context(), context()
    left["attachments"] = ["a"]
    right["attachments"] = ["b"]
    assert digest_context(left) == digest_context(right)
