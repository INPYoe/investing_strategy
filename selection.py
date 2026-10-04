"""선정점수 계산(S = 50×I + 30×L + 20×C) 및 집중 제한 적용 종목 선정. 주문은 만들지 않는다."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from guru_tracker.study import ROOT

TARGET_WEIGHT_PCT = 12.5
MAX_HOLDINGS = 8
SECTOR_CAP_PCT = 30.0
GROUP_CAP_PCT = 40.0


def compute_scores(consecutive_counts: pd.DataFrame) -> pd.DataFrame:
    """I·L·C·S 점수를 계산해 적격 종목 테이블을 반환한다.

    입력: multi_guru.build_consecutive_counts() 출력 또는 consecutive_counts.csv.
    출력: meets_entry_condition이 True인 CUSIP 하나당 한 행, S 내림차순 정렬.

    점수 공식 (master doc §5):
      I = mean(min(w_mj / 0.20, 1))   w_mj = 운용사 m에서 기업 j의 비중 (0~1)
      L = mean(min(n_mj / 12, 1))     n_mj = 연속 보유 공시 수
      C = k_j / M                     k_j = 적격 운용사 수, M = 전체 운용사 수
      S = 50×I + 30×L + 20×C
    """
    columns = ["cusip", "symbol", "issuer", "I", "L", "C", "S",
               "qualifying_manager_count", "M", "qualifying_managers", "max_consecutive_quarters"]
    if consecutive_counts.empty or "meets_entry_condition" not in consecutive_counts.columns:
        return pd.DataFrame(columns=columns)
    eligible = consecutive_counts.loc[consecutive_counts.meets_entry_condition].copy()
    if eligible.empty:
        return pd.DataFrame(columns=columns)

    M = int(eligible["M"].iloc[0])
    rows = []
    for cusip, grp in eligible.groupby("cusip"):
        k_j = len(grp)
        # w_mj는 current_weight_pct(%) → 분율(0~1)로 변환 후 20% 포화
        I_j = grp["current_weight_pct"].apply(lambda w: min(w / 100 / 0.20, 1.0)).mean()
        L_j = grp["consecutive_quarters"].apply(lambda n: min(n / 12, 1.0)).mean()
        C_j = k_j / M
        S_j = 50 * I_j + 30 * L_j + 20 * C_j
        sym = grp["symbol"].mode()
        iss = grp["issuer"].mode()
        rows.append({
            "cusip": cusip,
            "symbol": sym.iloc[0] if not sym.empty else grp["symbol"].iloc[0],
            "issuer": iss.iloc[0] if not iss.empty else grp["issuer"].iloc[0],
            "I": round(I_j, 4),
            "L": round(L_j, 4),
            "C": round(C_j, 4),
            "S": round(S_j, 4),
            "qualifying_manager_count": k_j,
            "M": M,
            "qualifying_managers": "|".join(sorted(grp["manager_id"].tolist())),
            "max_consecutive_quarters": int(grp["consecutive_quarters"].max()),
        })

    df = pd.DataFrame(rows, columns=columns)
    # 동점 정렬: S → 적격 운용사 수 → 최장 연속 → CUSIP 사전순
    return df.sort_values(
        ["S", "qualifying_manager_count", "max_consecutive_quarters", "cusip"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def select_portfolio(
    scores: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
    group_map: dict[str, list[str]] | None = None,
    max_holdings: int = MAX_HOLDINGS,
    target_weight_pct: float = TARGET_WEIGHT_PCT,
    sector_cap_pct: float = SECTOR_CAP_PCT,
    group_cap_pct: float = GROUP_CAP_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """점수 순서대로 종목을 선택하며 섹터·그룹 집중 제한을 적용한다.

    반환: (selected_df, skipped_df)
      selected_df: 선정된 종목, target_weight_pct 열 포함.
      skipped_df:  집중 제한으로 건너뛴 종목과 건너뛴 이유.
    """
    if sector_map is None:
        sector_map = {}
    if group_map is None:
        group_map = {}

    selected: list[dict] = []
    skipped: list[dict] = []
    sector_weight: dict[str, float] = {}
    group_weight: dict[str, float] = {}

    for _, row in scores.iterrows():
        if len(selected) >= max_holdings:
            break
        cusip = row["cusip"]
        sector = sector_map.get(cusip, "UNCLASSIFIED")
        groups = group_map.get(cusip, [])

        reasons: list[str] = []
        # 섹터 집중 확인 (UNCLASSIFIED는 무제한)
        if sector != "UNCLASSIFIED":
            if sector_weight.get(sector, 0.0) + target_weight_pct > sector_cap_pct + 1e-9:
                reasons.append(f"sector_cap:{sector}")
        # 그룹 집중 확인 (복수 그룹 가능)
        for gid in groups:
            if group_weight.get(gid, 0.0) + target_weight_pct > group_cap_pct + 1e-9:
                reasons.append(f"group_cap:{gid}")

        if reasons:
            skipped.append({**row.to_dict(), "sector": sector, "groups": "|".join(groups),
                            "skip_reason": "|".join(reasons)})
            continue

        selected.append({
            **row.to_dict(),
            "target_weight_pct": target_weight_pct,
            "sector": sector,
            "groups": "|".join(groups),
        })
        if sector != "UNCLASSIFIED":
            sector_weight[sector] = sector_weight.get(sector, 0.0) + target_weight_pct
        for gid in groups:
            group_weight[gid] = group_weight.get(gid, 0.0) + target_weight_pct

    selected_df = pd.DataFrame(selected) if selected else pd.DataFrame()
    skipped_df = pd.DataFrame(skipped) if skipped else pd.DataFrame()
    return selected_df, skipped_df


def run(asof: str,
        sector_map: dict[str, str] | None = None,
        group_map: dict[str, list[str]] | None = None) -> Path:
    """consecutive_counts.csv를 읽어 점수를 계산하고 포트폴리오를 선정한다."""
    counts_path = ROOT / "reports" / asof / "multi_guru" / "consecutive_counts.csv"
    if not counts_path.exists():
        raise FileNotFoundError(f"CONSECUTIVE_COUNTS_MISSING:{counts_path}")

    counts_df = pd.read_csv(counts_path)

    # sector_map·group_map 미제공 시 sector_groups.json의 tickers 배열에서 빌드
    # (종목 선정 전에는 tickers가 비어 있어 모두 UNCLASSIFIED·빈 그룹이 된다)
    if sector_map is None and group_map is None:
        sg = json.loads((ROOT / "sector_groups.json").read_text())
        sector_map = {}
        group_map = {}
        for grp in sg.get("related_groups", []):
            for ticker in grp.get("tickers", []):
                group_map.setdefault(ticker, []).append(grp["id"])

    output_dir = ROOT / "reports" / asof / "selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = compute_scores(counts_df)
    selected_df, skipped_df = select_portfolio(scores, sector_map=sector_map, group_map=group_map)

    scores.to_csv(output_dir / "scores.csv", index=False)
    selected_df.to_csv(output_dir / "portfolio.csv", index=False)
    skipped_df.to_csv(output_dir / "skipped_candidates.csv", index=False)

    n_eligible = len(scores)
    n_selected = len(selected_df)
    cash_pct = (MAX_HOLDINGS - n_selected) * TARGET_WEIGHT_PCT

    lines = [
        "# 장기 핵심 종목 선정 결과", "",
        f"기준일: {asof}. 적격 후보: {n_eligible}개. 선정: {n_selected}개. 현금: {cash_pct:.1f}%.", "",
        "## 선정 종목 (목표비중 12.5%)", "",
        "| 순위 | 종목 | CUSIP | S | I | L | C | 적격 운용사 | 연속 분기 | 섹터 | 그룹 |",
        "|---:|---|---|---:|---:|---:|---:|---|---:|---|---|",
    ]
    if not selected_df.empty:
        for i, r in enumerate(selected_df.itertuples(), start=1):
            lines.append(
                f"| {i} | {r.symbol} | {r.cusip} | {r.S:.1f}"
                f" | {r.I:.3f} | {r.L:.3f} | {r.C:.3f}"
                f" | {r.qualifying_managers} | {r.max_consecutive_quarters}"
                f" | {r.sector} | {r.groups} |"
            )
    else:
        lines.append("| — | — | — | — | — | — | — | — | — | — | — |")

    if not skipped_df.empty:
        lines += [
            "", "## 집중 제한으로 건너뛴 후보", "",
            "| 종목 | CUSIP | S | 건너뛴 이유 |", "|---|---|---:|---|",
        ]
        for r in skipped_df.itertuples():
            lines.append(f"| {r.symbol} | {r.cusip} | {r.S:.1f} | {r.skip_reason} |")

    lines += [
        "", "## 적격 후보 전체 점수 (S 내림차순)", "",
        "| 순위 | 종목 | CUSIP | S | I | L | C | 적격 운용사 수 | 최장 연속 분기 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(scores.itertuples(), start=1):
        lines.append(
            f"| {i} | {r.symbol} | {r.cusip} | {r.S:.1f}"
            f" | {r.I:.3f} | {r.L:.3f} | {r.C:.3f}"
            f" | {r.qualifying_manager_count} | {r.max_consecutive_quarters} |"
        )

    if cash_pct > 0:
        lines += [
            "", f"## 현금 유보 {cash_pct:.1f}%", "",
            "적격 후보 부족 또는 집중 제한으로 빈 자리에 현금을 유지한다. 나머지 기업에 재배분하지 않는다.",
        ]

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    metadata = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "asof": asof,
        "eligible_candidates": n_eligible,
        "selected_count": n_selected,
        "cash_pct": cash_pct,
        "target_weight_pct": TARGET_WEIGHT_PCT,
        "sector_cap_pct": SECTOR_CAP_PCT,
        "group_cap_pct": GROUP_CAP_PCT,
        "sector_map_size": len(sector_map) if sector_map else 0,
        "group_map_size": len(group_map) if group_map else 0,
        "orders_sent": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="선정점수 계산 및 종목 선정. 주문 없음.")
    parser.add_argument("--asof", required=True, help="YYYY-MM-DD 기준일")
    args = parser.parse_args()
    out = run(args.asof)
    print(out)
