from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize


YEARS = np.arange(2026, 2036)


def simulate(shares: np.ndarray, shock: bool = False) -> pd.DataFrame:
    K, D, AI, H, A, L = 27500.0, 20.3, 86.0, 30.0, 34.913621, 53.9
    rows = []
    for t, year in enumerate(YEARS):
        Y = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
        if shock and year == 2028:
            Y *= 0.92
        invest = 0.24 * Y
        s = shares[t]
        C = max(Y - invest, 1.0)
        rows.append({"year": year, "K": K, "D": D, "AI": AI, "H": H, "A": A, "Y": Y, "C": C, "I_K": invest * s[0], "I_D": invest * s[1], "I_AI": invest * s[2], "I_H": invest * s[3]})
        K = 0.95 * K + invest * s[0]
        D = 0.88 * D + invest * s[1] / 100
        AI = 0.85 * AI + invest * s[2] / 20
        H = H + 0.8 * invest * s[3] / 200 - 0.02 * H
        A = A * (1 + 0.003 * D / 100 + 0.002 * AI / 100 + 0.004 * H / 100)
    return pd.DataFrame(rows)


def welfare(shares_flat: np.ndarray) -> float:
    raw = shares_flat.reshape(len(YEARS), 4)
    shares = raw / raw.sum(axis=1, keepdims=True)
    df = simulate(shares)
    return -sum((0.97**t) * np.log(max(c, 1.0)) for t, c in enumerate(df["C"]))


def optimize_path() -> np.ndarray:
    x0 = np.tile([0.35, 0.25, 0.20, 0.20], len(YEARS))
    bounds = [(0.05, 0.70)] * len(x0)
    res = minimize(welfare, x0, method="SLSQP", bounds=bounds, options={"maxiter": 300, "ftol": 1e-8})
    raw = res.x.reshape(len(YEARS), 4)
    return raw / raw.sum(axis=1, keepdims=True)


def save_path_plot(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for col in ["K", "D", "AI", "H", "Y", "C"]:
        axes[0].plot(df["year"], df[col], marker="o", label=col)
    for col in ["I_K", "I_D", "I_AI", "I_H"]:
        axes[1].plot(df["year"], df[col], marker="o", label=col)
    axes[0].legend(ncol=3)
    axes[1].legend(ncol=4)
    axes[0].set_title("Exercise 8 - Dynamic optimal trajectories")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    opt_shares = optimize_path()
    opt = simulate(opt_shares)
    shock = simulate(opt_shares, shock=True)
    even = simulate(np.tile([0.25, 0.25, 0.25, 0.25], (len(YEARS), 1)))
    front = simulate(np.array([[0.45, 0.25, 0.15, 0.15] if i < 3 else [0.25, 0.25, 0.25, 0.25] for i in range(len(YEARS))]))
    opt.to_csv(out / "ex8_dynamic_optimal_path.csv", index=False)
    shock.to_csv(out / "ex8_dynamic_shock_path.csv", index=False)
    pd.DataFrame(opt_shares, columns=["share_K", "share_D", "share_AI", "share_H"]).assign(year=YEARS).to_csv(out / "ex8_optimal_investment_shares.csv", index=False)
    pd.DataFrame([
        {"strategy": "optimal", "welfare": -welfare(opt_shares.ravel()), "Y_2035": opt["Y"].iloc[-1]},
        {"strategy": "even", "welfare": sum((0.97**i) * np.log(c) for i, c in enumerate(even["C"])), "Y_2035": even["Y"].iloc[-1]},
        {"strategy": "front_load", "welfare": sum((0.97**i) * np.log(c) for i, c in enumerate(front["C"])), "Y_2035": front["Y"].iloc[-1]},
    ]).to_csv(out / "ex8_strategy_comparison.csv", index=False)
    save_path_plot(opt, out / "ex8_dynamic_trajectories.png")
    print("=== Exercise 8 Completed ===")
    print(opt.tail(1).to_string(index=False))


if __name__ == "__main__":
    main()
