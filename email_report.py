"""구루 전략 상태를 Gmail HTML 보고서로 발송한다. 주문 API는 사용하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import smtplib
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import dotenv_values

from guru_tracker.study import ROOT


STATE_PATH = ROOT / "data" / "automation" / "email_state.json"


def latest_complete_multi_guru() -> Path:
    candidates = sorted((ROOT / "reports").glob("*/multi_guru"), reverse=True)
    for path in candidates:
        metadata = path / "metadata.json"
        if metadata.exists() and json.loads(metadata.read_text()).get("status") == "COMPLETE":
            return path
    raise RuntimeError("NO_COMPLETE_MULTI_GURU_REPORT")


def report_context() -> dict:
    """최신 완료된 분기 공시 결과와 선정 포트폴리오를 이메일 컨텍스트로 반환한다."""
    report = latest_complete_multi_guru()
    asof = report.parent.name
    metadata = json.loads((report / "metadata.json").read_text())

    # 목표 포트폴리오: selection/portfolio.csv에서 12.5% 동일비중
    portfolio_path = ROOT / "reports" / asof / "selection" / "portfolio.csv"
    if portfolio_path.exists():
        portfolio_df = pd.read_csv(portfolio_path)
        targets = {row["symbol"]: float(row["target_weight_pct"])
                   for _, row in portfolio_df.iterrows()}
    else:
        targets = {}

    # 연속 보유 현황: consecutive_counts.csv에서 편입 조건 충족 종목
    counts_path = report / "consecutive_counts.csv"
    if counts_path.exists():
        counts_df = pd.read_csv(counts_path)
        eligible = counts_df.loc[counts_df.meets_entry_condition]
        per_cusip = (eligible.groupby("cusip")
                     .agg(symbol=("symbol", "first"),
                          qualifying_managers=("manager_id", "nunique"),
                          max_consecutive=("consecutive_quarters", "max"))
                     .reset_index())
        directions = [
            {"symbol": row["symbol"], "candidate_level": "ENTRY_ELIGIBLE",
             "meaningful_manager_count": row["qualifying_managers"],
             "positive_signal_count": row["max_consecutive"],
             "reducing_manager_count": 0, "direction_state": "MEETS_ENTRY_CONDITION"}
            for _, row in per_cusip.iterrows()
        ]
    else:
        directions = []

    # 계좌 상태: tomorrow_plan/plan.json이 있으면 사용, 없으면 기본값
    plan_path = ROOT / "reports" / asof / "tomorrow_plan" / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        liquidation_status = plan.get("status", "PLAN_NOT_EXECUTED")
        domestic_holdings = plan.get("domestic_liquidation_preview", [])
        domestic_pending = plan.get("pending_overseas_orders", 0)
        orderable = plan.get("current_integrated_orderable_usd", "0")
    else:
        liquidation_status = "PLAN_NOT_EXECUTED"
        domestic_holdings = []
        domestic_pending = 0
        orderable = "0"

    return {
        "report_date": asof,
        "managers_collected": metadata["managers_collected"],
        "managers_requested": metadata["managers_requested"],
        "liquidation_status": liquidation_status,
        "domestic_holdings": domestic_holdings,
        "domestic_pending_order_count": domestic_pending,
        "usd_integrated_orderable": orderable,
        "targets": targets,
        "directions": directions,
        "orders_sent": False,
        "attachments": [report / "consecutive_counts.csv", report / "signals.csv"],
    }


def render_html(context: dict) -> str:
    target_rows = "".join(
        f"<tr><td>{html.escape(symbol)}</td><td style='text-align:right'>{weight:.2f}%</td></tr>"
        for symbol, weight in context["targets"].items())
    direction_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in
                           ["symbol", "candidate_level", "meaningful_manager_count",
                            "positive_signal_count", "reducing_manager_count", "direction_state"]) + "</tr>"
        for row in context["directions"])
    return f"""<!doctype html><html><body style="font-family:-apple-system,BlinkMacSystemFont,Arial;color:#202124">
<h2>구루 전략 투자 진행 보고</h2>
<p>공시 기준일 {html.escape(context['report_date'])}, 구루 수집 {context['managers_collected']}/{context['managers_requested']}.</p>
<h3>계좌 상태</h3>
<ul><li>국내 청산: {html.escape(context['liquidation_status'])}</li>
<li>국내 보유종목: {len(context['domestic_holdings'])}개</li>
<li>국내 미체결: {context['domestic_pending_order_count']}건</li>
<li>통합증거금 미국주식 주문가능액: ${html.escape(context['usd_integrated_orderable'])}</li></ul>
<h3>현재 목표 비중</h3><table border="1" cellpadding="7" cellspacing="0"><tr><th>종목</th><th>비중</th></tr>{target_rows}</table>
<h3>다중 구루 방향성</h3><table border="1" cellpadding="7" cellspacing="0">
<tr><th>종목</th><th>등급</th><th>유효 구루</th><th>매집</th><th>축소</th><th>상태</th></tr>{direction_rows}</table>
<h3>다음 확인</h3><p>미국 정규장 전 KIS 실시간 호가, CUSIP, 보유·미체결, 통합증거금 주문가능금액을 다시 검증합니다. 검증 실패 또는 가격 상한 초과 시 주문 계획을 중단합니다.</p>
<p style="color:#666">이 메일은 상태 보고서이며 이 발송 과정에서 주문을 제출하지 않습니다.</p>
</body></html>"""


def digest_context(context: dict) -> str:
    stable = {key: value for key, value in context.items() if key != "attachments"}
    return hashlib.sha256(json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def send(*, force: bool = False) -> str:
    config = json.loads((ROOT / "email_config.json").read_text())
    if not config["enabled"]:
        return "EMAIL_DISABLED"
    context = report_context()
    digest = digest_context(context)
    if STATE_PATH.exists() and not force and json.loads(STATE_PATH.read_text()).get("last_digest") == digest:
        return "UNCHANGED_NOT_SENT"
    credentials = dotenv_values(config["credential_file"])
    sender, password = credentials.get(config["sender_env"]), credentials.get(config["password_env"])
    if not sender or not password:
        raise RuntimeError("EMAIL_CREDENTIALS_MISSING")
    rendered = render_html(context)
    output = ROOT / "reports" / datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat() / "email"
    output.mkdir(parents=True, exist_ok=True)
    (output / "guru_status.html").write_text(rendered, encoding="utf-8")
    message = EmailMessage()
    message["Subject"] = f"{config['subject_prefix']} {datetime.now(ZoneInfo('Asia/Seoul')):%Y-%m-%d} 투자 진행 보고"
    message["From"], message["To"] = sender, config["recipient"]
    message.set_content("구루 전략 투자 진행 보고입니다. HTML 보기를 지원하는 메일 앱에서 확인하세요.")
    message.add_alternative(rendered, subtype="html")
    for path in context["attachments"]:
        message.add_attachment(path.read_bytes(), maintype="application", subtype="octet-stream", filename=path.name)
    with smtplib.SMTP_SSL(config["smtp_host"], int(config["smtp_port"]), timeout=30) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)
    STATE_PATH.write_text(json.dumps({"last_digest": digest, "sent_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
                                      "recipient": config["recipient"]}, ensure_ascii=False, indent=2))
    return "SENT"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="구루 전략 이메일 보고")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.send:
        print(send(force=args.force))
    else:
        context = report_context()
        output = ROOT / "reports" / datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat() / "email"
        output.mkdir(parents=True, exist_ok=True)
        (output / "guru_status.html").write_text(render_html(context), encoding="utf-8")
        print(output / "guru_status.html")
