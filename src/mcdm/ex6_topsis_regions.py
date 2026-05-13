from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CRITERIA = [
    "grdp_per_capita_million_VND",
    "fdi_registered_billion_USD",
    "digital_index_0_100",
    "ai_readiness_0_100",
    "trained_labor_pct",
    "rd_intensity_pct",
    "internet_penetration_pct",
    "gini_coef",
]
IS_BENEFIT = np.array([True, True, True, True, True, True, True, False])
EXPERT_WEIGHTS = np.array([0.10, 0.10, 0.15, 0.20, 0.15, 0.15, 0.05, 0.10])


def topsis(df: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    x = df[CRITERIA].to_numpy(float)
    r = x / np.sqrt((x**2).sum(axis=0))
    v = r * weights
    a_star = np.where(IS_BENEFIT, v.max(axis=0), v.min(axis=0))
    a_neg = np.where(IS_BENEFIT, v.min(axis=0), v.max(axis=0))
    s_star = np.sqrt(((v - a_star) ** 2).sum(axis=1))
    s_neg = np.sqrt(((v - a_neg) ** 2).sum(axis=1))
    out = df[["region_code", "region_name_vi"]].copy()
    out["TOPSIS_score"] = s_neg / (s_star + s_neg)
    return out.sort_values("TOPSIS_score", ascending=False).reset_index(drop=True)


def entropy_weights(x: np.ndarray) -> np.ndarray:
    x = x.copy().astype(float)
    for j, benefit in enumerate(IS_BENEFIT):
        if not benefit:
            x[:, j] = x[:, j].max() - x[:, j] + 1e-9
    p = x / x.sum(axis=0)
    e = -(1 / np.log(len(x))) * np.sum(p * np.log(p + 1e-12), axis=0)
    d = 1 - e
    return d / d.sum()


def sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base = EXPERT_WEIGHTS.copy()
    for ai_w in np.arange(0.10, 0.401, 0.05):
        w = base.copy()
        w[3] = ai_w
        w = w / w.sum()
        ranked = topsis(df, w)
        for rank, row in ranked.iterrows():
            rows.append({"ai_weight": round(ai_w, 2), "rank": rank + 1, **row.to_dict()})
    return pd.DataFrame(rows)


def save_bar(rank: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.bar(rank["region_code"], rank["TOPSIS_score"])
    plt.title("Exercise 6 - TOPSIS region scores")
    plt.xlabel("Region")
    plt.ylabel("C*")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    df = pd.read_csv(root / "data" / "vietnam_regions_2024.csv")
    expert = topsis(df, EXPERT_WEIGHTS)
    ew = entropy_weights(df[CRITERIA].to_numpy(float))
    entropy = topsis(df, ew)
    sens = sensitivity(df)

    expert.to_csv(out / "ex6_topsis_expert_rank.csv", index=False)
    entropy.to_csv(out / "ex6_topsis_entropy_rank.csv", index=False)
    pd.DataFrame({"criterion": CRITERIA, "entropy_weight": ew}).to_csv(out / "ex6_entropy_weights.csv", index=False)
    sens.to_csv(out / "ex6_ai_weight_sensitivity.csv", index=False)
    save_bar(expert, out / "ex6_topsis_expert_scores.png")

    print("=== Exercise 6 Completed ===")
    print("Expert weights:")
    print(expert.to_string(index=False))
    print("\nEntropy weights:")
    print(entropy.to_string(index=False))


if __name__ == "__main__":
    main()
