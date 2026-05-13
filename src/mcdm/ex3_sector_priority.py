from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


GOOD_COLS = [
    "growth_rate_2024_pct",
    "productivity_million_VND_per_worker",
    "spillover_coef_0_1",
    "export_billion_USD",
    "labor_million",
    "ai_readiness_0_100",
]
BAD_COL = "automation_risk_pct"
WEIGHTS = np.array([0.15, 0.15, 0.20, 0.15, 0.10, 0.20])
RISK_WEIGHT = 0.15


def norm_good(x: pd.Series) -> pd.Series:
    span = x.max() - x.min()
    return (x - x.min()) / span if span else x * 0


def norm_bad(x: pd.Series) -> pd.Series:
    span = x.max() - x.min()
    return (x.max() - x) / span if span else x * 0


def normalized_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["sector_name_vi"]].copy()
    out[GOOD_COLS] = df[GOOD_COLS].apply(norm_good)
    out["automation_risk_score"] = norm_bad(df[BAD_COL])
    return out


def rank_with_weights(df: pd.DataFrame, weights: np.ndarray, risk_weight: float) -> pd.DataFrame:
    x = normalized_matrix(df)
    score = x[GOOD_COLS].to_numpy(float) @ weights - risk_weight * x["automation_risk_score"].to_numpy(float)
    out = df[["sector_name_vi"]].copy()
    out["Priority"] = score
    return out.sort_values("Priority", ascending=False).reset_index(drop=True)


def ai_weight_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = np.r_[WEIGHTS, RISK_WEIGHT]
    for ai_w in np.arange(0.05, 0.401, 0.05):
        w = base.copy()
        w[5] = ai_w
        w = w / w.sum()
        rank = rank_with_weights(df, w[:6], w[6])
        for pos, row in rank.reset_index().iterrows():
            rows.append({"ai_weight": round(ai_w, 2), "sector_name_vi": row["sector_name_vi"], "rank": pos + 1})
    return pd.DataFrame(rows)


def save_heatmap(sens: pd.DataFrame, path: Path) -> None:
    pivot = sens.pivot(index="sector_name_vi", columns="ai_weight", values="rank")
    plt.figure(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis_r", cbar_kws={"label": "Rank"})
    plt.title("Exercise 3 - Top rank sensitivity to AI readiness weight")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(root / "data" / "vietnam_sectors_2024.csv")

    normalized = normalized_matrix(df)
    default_rank = rank_with_weights(df, WEIGHTS, RISK_WEIGHT)
    priorities = pd.read_csv(root / "data" / "vietnam_priorities.csv")
    comparisons = []
    for _, p in priorities.iterrows():
        w = p[["weight_growth", "weight_productivity", "weight_spillover", "weight_export", "weight_employment", "weight_ai_readiness"]].to_numpy(float)
        risk_w = float(p["weight_risk"])
        ranked = rank_with_weights(df, w, risk_w)
        for rank, row in ranked.head(3).iterrows():
            comparisons.append({"profile": p["profile"], "rank": rank + 1, **row.to_dict()})
    comparison_df = pd.DataFrame(comparisons)
    sens = ai_weight_sensitivity(df)

    normalized.to_csv(out / "ex3_normalized_matrix.csv", index=False)
    default_rank.to_csv(out / "ex3_sector_priority_rank.csv", index=False)
    comparison_df.to_csv(out / "ex3_policy_weight_top3.csv", index=False)
    sens.to_csv(out / "ex3_ai_weight_sensitivity.csv", index=False)
    save_heatmap(sens, out / "ex3_ai_weight_sensitivity_heatmap.png")

    print("=== Exercise 3 Completed ===")
    print(default_rank.to_string(index=False))
    print("\nTop-3 by policy profile:")
    print(comparison_df.to_string(index=False))


if __name__ == "__main__":
    main()
