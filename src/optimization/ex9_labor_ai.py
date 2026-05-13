from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp


SECTORS = ["Nong-Lam-Thuy san", "CN che bien che tao", "Xay dung", "Ban buon-ban le", "Tai chinh-Ngan hang", "Logistics-Van tai", "CNTT-Truyen thong", "Giao duc-Dao tao"]
LABOR = np.array([13.20, 11.50, 4.80, 7.80, 0.55, 1.95, 0.62, 2.15])
RISK = np.array([18, 42, 25, 38, 52, 35, 28, 22]) / 100
A1 = np.array([8.5, 32.5, 12.8, 22.4, 45.8, 28.5, 62.5, 18.5])
B1 = np.array([45, 28, 35, 32, 22, 30, 20, 55])
C1 = np.array([5.2, 62.4, 18.5, 48.2, 72.5, 42.8, 32.5, 12.5])
D1 = np.array([50, 32, 42, 38, 26, 36, 24, 62])


def solve_labor(limit_displacement: bool = False) -> tuple[pd.DataFrame, str, float]:
    m = pulp.LpProblem("VN_AI_Labor", pulp.LpMaximize)
    x_ai = pulp.LpVariable.dicts("x_AI", range(8), lowBound=0)
    x_h = pulp.LpVariable.dicts("x_H", range(8), lowBound=0)
    net = {i: (A1[i] - C1[i] * RISK[i]) * x_ai[i] + B1[i] * x_h[i] for i in range(8)}
    displaced = {i: C1[i] * RISK[i] * x_ai[i] for i in range(8)}
    retrain = {i: D1[i] * x_h[i] for i in range(8)}
    m += pulp.lpSum(net[i] for i in range(8))
    m += pulp.lpSum(x_ai[i] + x_h[i] for i in range(8)) <= 30000
    for i in range(8):
        m += net[i] >= 0
        m += displaced[i] <= retrain[i]
        if limit_displacement:
            m += displaced[i] <= 0.05 * LABOR[i] * 1_000_000
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status].lower()
    rows = []
    for i, name in enumerate(SECTORS):
        ai = float(pulp.value(x_ai[i]) or 0)
        h = float(pulp.value(x_h[i]) or 0)
        rows.append({"sector_name_vi": name, "x_AI": ai, "x_H": h, "NewJob": A1[i] * ai, "UpgradeJob": B1[i] * h, "DisplacedJob": C1[i] * RISK[i] * ai, "RetrainingCapacity": D1[i] * h, "NetJob": (A1[i] - C1[i] * RISK[i]) * ai + B1[i] * h})
    return pd.DataFrame(rows), status, float(pulp.value(m.objective) or 0)


def save_flow_plot(df: pd.DataFrame, path: Path) -> None:
    vulnerable = df[df["sector_name_vi"].isin(["Nong-Lam-Thuy san", "Xay dung", "Ban buon-ban le"])]
    y = np.arange(len(vulnerable))
    plt.figure(figsize=(8, 4))
    plt.barh(y - 0.2, vulnerable["DisplacedJob"], height=0.35, label="Displaced")
    plt.barh(y + 0.2, vulnerable["RetrainingCapacity"], height=0.35, label="Retraining capacity")
    plt.yticks(y, vulnerable["sector_name_vi"])
    plt.title("Exercise 9 - Vulnerable labor transition capacity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    base, status, z = solve_labor(False)
    limited, lim_status, lim_z = solve_labor(True)
    base.to_csv(out / "ex9_labor_allocation.csv", index=False)
    limited.to_csv(out / "ex9_labor_displacement_limited.csv", index=False)
    x_ai_max = 30000
    min_h_sector2 = (C1[1] * RISK[1] - A1[1]) * x_ai_max / B1[1]
    pd.DataFrame([{"case": "base", "status": status, "total_NetJob": z}, {"case": "displacement_limit", "status": lim_status, "total_NetJob": lim_z}, {"case": "sector2_min_H_if_all_AI", "status": "formula", "total_NetJob": min_h_sector2}]).to_csv(out / "ex9_summary.csv", index=False)
    save_flow_plot(base, out / "ex9_vulnerable_labor_flow.png")
    print("=== Exercise 9 Completed ===")
    print(status, round(z, 2))
    print(base.to_string(index=False))


if __name__ == "__main__":
    main()
