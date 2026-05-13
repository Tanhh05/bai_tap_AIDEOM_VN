from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Elasticities:
    alpha: float = 0.33
    beta: float = 0.42
    gamma: float = 0.10
    delta: float = 0.08
    theta: float = 0.07

    def validate(self) -> None:
        s = self.alpha + self.beta + self.gamma + self.delta + self.theta
        if not np.isclose(s, 1.0, atol=1e-6):
            raise ValueError(f"Elasticities must sum to 1.0, got {s:.6f}")


def load_macro_data(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).sort_values("year").reset_index(drop=True)
    required = [
        "year",
        "GDP_trillion_VND",
        "K_trillion_VND",
        "L_million_workers",
        "D_digital_pct",
        "AI_thousand_firms",
        "H_trained_labor_pct",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def estimate_tfp(df: pd.DataFrame, e: Elasticities) -> np.ndarray:
    y = df["GDP_trillion_VND"].to_numpy(float)
    k = df["K_trillion_VND"].to_numpy(float)
    l = df["L_million_workers"].to_numpy(float)
    d = df["D_digital_pct"].to_numpy(float)
    ai = df["AI_thousand_firms"].to_numpy(float)
    h = df["H_trained_labor_pct"].to_numpy(float)
    return y / (k**e.alpha * l**e.beta * d**e.gamma * ai**e.delta * h**e.theta)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)


def growth_decomposition(df: pd.DataFrame, e: Elasticities) -> pd.DataFrame:
    out = pd.DataFrame({"year": df["year"]})
    out["Y"] = df["GDP_trillion_VND"]
    out["K"] = df["K_trillion_VND"]
    out["L"] = df["L_million_workers"]
    out["D"] = df["D_digital_pct"]
    out["AI"] = df["AI_thousand_firms"]
    out["H"] = df["H_trained_labor_pct"]

    for c in ["Y", "K", "L", "D", "AI", "H"]:
        out[f"ln_{c}"] = np.log(out[c].astype(float))
        out[f"dln_{c}"] = out[f"ln_{c}"].diff()

    out["contrib_K"] = e.alpha * out["dln_K"]
    out["contrib_L"] = e.beta * out["dln_L"]
    out["contrib_D"] = e.gamma * out["dln_D"]
    out["contrib_AI"] = e.delta * out["dln_AI"]
    out["contrib_H"] = e.theta * out["dln_H"]
    out["dln_A"] = out["dln_Y"] - (
        out["contrib_K"] + out["contrib_L"] + out["contrib_D"] + out["contrib_AI"] + out["contrib_H"]
    )
    return out


def simulate_2030(df: pd.DataFrame, e: Elasticities, tfp_growth: float = 0.012) -> float:
    row_2025 = df[df["year"] == 2025].iloc[0]
    a_2025 = estimate_tfp(df[df["year"] == 2025], e)[0]

    years = 5  # 2026..2030
    k_2030 = float(row_2025["K_trillion_VND"]) * (1.06**years)
    l_2030 = float(row_2025["L_million_workers"]) * (1.06**years)
    d_2030 = 30.0
    ai_2030 = 100.0
    h_2030 = 35.0
    a_2030 = a_2025 * ((1.0 + tfp_growth) ** years)

    y_2030 = a_2030 * (k_2030**e.alpha) * (l_2030**e.beta) * (d_2030**e.gamma) * (ai_2030**e.delta) * (h_2030**e.theta)
    return float(y_2030)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    e = Elasticities()
    e.validate()

    df = load_macro_data(root / "data" / "vietnam_macro_2020_2025.csv")
    df["A_t"] = estimate_tfp(df, e)

    # 1.4.2
    a_bar = float(df["A_t"].mean())
    y_hat = a_bar * (
        df["K_trillion_VND"] ** e.alpha
        * df["L_million_workers"] ** e.beta
        * df["D_digital_pct"] ** e.gamma
        * df["AI_thousand_firms"] ** e.delta
        * df["H_trained_labor_pct"] ** e.theta
    )
    df["Y_hat_from_A_bar"] = y_hat
    mape_val = mape(df["GDP_trillion_VND"].to_numpy(float), y_hat.to_numpy(float))

    # 1.4.3
    decomp = growth_decomposition(df, e)
    avg = decomp[["contrib_K", "contrib_L", "contrib_D", "contrib_AI", "contrib_H", "dln_A"]].iloc[1:].mean()
    total = float(avg.sum())
    share_pct = (avg / total * 100.0).rename("share_pct")
    share_df = share_pct.reset_index().rename(columns={"index": "component"})

    # 1.4.4
    y_2030 = simulate_2030(df, e, tfp_growth=0.012)

    # Save tables
    df.to_csv(out_dir / "ex1_tfp_and_prediction.csv", index=False)
    decomp.to_csv(out_dir / "ex1_growth_decomposition.csv", index=False)
    share_df.to_csv(out_dir / "ex1_growth_contribution_share.csv", index=False)

    # Plot A_t
    plt.figure(figsize=(8, 4))
    plt.plot(df["year"], df["A_t"], marker="o")
    plt.title("Exercise 1 - Estimated TFP (A_t), Vietnam 2020-2025")
    plt.xlabel("Year")
    plt.ylabel("A_t")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ex1_tfp_trend.png", dpi=150)
    plt.close()

    # Plot growth contribution
    plt.figure(figsize=(8, 4))
    plt.bar(share_df["component"], share_df["share_pct"])
    plt.title("Exercise 1 - Average Growth Contribution Share (2021-2025)")
    plt.ylabel("Share (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_dir / "ex1_growth_contribution_share.png", dpi=150)
    plt.close()

    print("=== Exercise 1 Completed ===")
    print("Saved:")
    print("-", out_dir / "ex1_tfp_and_prediction.csv")
    print("-", out_dir / "ex1_growth_decomposition.csv")
    print("-", out_dir / "ex1_growth_contribution_share.csv")
    print("-", out_dir / "ex1_tfp_trend.png")
    print("-", out_dir / "ex1_growth_contribution_share.png")
    print(f"MAPE (Y vs Y_hat using mean A): {mape_val:.4f}%")
    print(f"Simulated GDP 2030 (trillion VND): {y_2030:.2f}")
    print("\nA_t by year:")
    print(df[["year", "A_t"]].to_string(index=False))
    print("\nAverage contribution shares (%):")
    print(share_df.to_string(index=False))


if __name__ == "__main__":
    main()
