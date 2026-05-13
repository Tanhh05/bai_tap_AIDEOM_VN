from __future__ import annotations

from pathlib import Path

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns


ITEMS = ["I", "D", "AI", "H"]


def load_region_data(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "vietnam_regions_2024.csv")


def solve_pulp(df: pd.DataFrame, fairness: bool = True, allow_slack: bool = False) -> tuple[pd.DataFrame, float, str]:
    regions = df["region_code"].tolist()
    beta = {(r.region_code, j): float(getattr(r, f"impact_{j}")) for r in df.itertuples() for j in ITEMS}
    d0 = dict(zip(df["region_code"], df["digital_index_0_100"]))
    gamma, lam = 0.002, 0.7

    m = pulp.LpProblem("VN_Digital_Budget_Regional", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", (regions, ITEMS), lowBound=0)
    slack = pulp.LpVariable.dicts("fairness_slack", regions, lowBound=0) if allow_slack else None
    penalty = 100000.0
    m += (
        pulp.lpSum(beta[(r, j)] * x[r][j] for r in regions for j in ITEMS)
        - (penalty * pulp.lpSum(slack[r] for r in regions) if allow_slack else 0)
    )
    m += pulp.lpSum(x[r][j] for r in regions for j in ITEMS) <= 50000, "budget_total"
    for r in regions:
        m += pulp.lpSum(x[r][j] for j in ITEMS) >= 5000, f"floor_{r}"
        m += pulp.lpSum(x[r][j] for j in ITEMS) <= 12000, f"cap_{r}"
    m += pulp.lpSum(x[r]["H"] for r in regions) >= 12000, "human_capital_floor"
    if fairness:
        M = pulp.LpVariable("Dmax", lowBound=0)
        for r in regions:
            m += d0[r] + gamma * x[r]["D"] <= M, f"dmax_{r}"
            if allow_slack:
                m += d0[r] + gamma * x[r]["D"] + slack[r] >= lam * M, f"fairness_{r}"
            else:
                m += d0[r] + gamma * x[r]["D"] >= lam * M, f"fairness_{r}"
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[m.status].lower()
    if status != "optimal":
        return pd.DataFrame(), float("nan"), status

    rows = []
    for r in regions:
        row = {"region_code": r, "region_name_vi": df.loc[df["region_code"] == r, "region_name_vi"].iloc[0]}
        row.update({j: float(pulp.value(x[r][j])) for j in ITEMS})
        row["total"] = sum(row[j] for j in ITEMS)
        if allow_slack:
            row["fairness_slack"] = float(pulp.value(slack[r]))
        rows.append(row)
    raw_z = sum(beta[(r, j)] * float(pulp.value(x[r][j])) for r in regions for j in ITEMS)
    return pd.DataFrame(rows), float(raw_z), status


def solve_cvxpy(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    beta = df[[f"impact_{j}" for j in ITEMS]].to_numpy(float)
    d0 = df["digital_index_0_100"].to_numpy(float)
    x = cp.Variable((len(df), len(ITEMS)), nonneg=True)
    M = cp.Variable(nonneg=True)
    constraints = [
        cp.sum(x) <= 50000,
        cp.sum(x, axis=1) >= 5000,
        cp.sum(x, axis=1) <= 12000,
        cp.sum(x[:, 3]) >= 12000,
        d0 + 0.002 * x[:, 1] <= M,
        d0 + 0.002 * x[:, 1] >= 0.7 * M,
    ]
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(beta, x))), constraints)
    prob.solve(solver=cp.CLARABEL)
    out = pd.DataFrame(x.value, columns=ITEMS)
    out.insert(0, "region_name_vi", df["region_name_vi"])
    out.insert(0, "region_code", df["region_code"])
    out["total"] = out[ITEMS].sum(axis=1)
    return out, float(prob.value)


def save_heatmap(alloc: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 4))
    sns.heatmap(alloc.set_index("region_code")[ITEMS], annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Exercise 4 - Optimal allocation by region and item")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    df = load_region_data(root)
    hard_alloc, hard_z, hard_status = solve_pulp(df, fairness=True)
    if hard_status == "optimal":
        alloc, z, fair_status = hard_alloc, hard_z, hard_status
    else:
        alloc, z, fair_status = solve_pulp(df, fairness=True, allow_slack=True)
    try:
        cvx_alloc, cvx_z = solve_cvxpy(df)
        cvx_status = "optimal" if np.isfinite(cvx_z) else "infeasible"
    except Exception:
        cvx_alloc, cvx_z, cvx_status = pd.DataFrame(), float("nan"), "solver_error"
    nofair, nofair_z, nofair_status = solve_pulp(df, fairness=False)

    alloc.to_csv(out / "ex4_pulp_allocation.csv", index=False)
    if not cvx_alloc.empty:
        cvx_alloc.to_csv(out / "ex4_cvxpy_allocation.csv", index=False)
    nofair.to_csv(out / "ex4_no_fairness_allocation.csv", index=False)
    pd.DataFrame(
        [
            {"model": "fairness_hard", "status": hard_status, "Z": hard_z},
            {"model": "fairness_with_slack", "status": fair_status, "Z": z},
            {"model": "no_fairness", "status": nofair_status, "Z": nofair_z},
            {"model": "cvxpy_hard_fairness", "status": cvx_status, "Z": cvx_z},
        ]
    ).to_csv(out / "ex4_objective_comparison.csv", index=False)
    save_heatmap(alloc, out / "ex4_allocation_heatmap.png")

    print("=== Exercise 4 Completed ===")
    print("Hard fairness status:", hard_status)
    print("Reported fairness allocation status:", fair_status, "Z*", round(z, 2))
    print("CVXPY hard fairness status:", cvx_status, "No fairness Z*", round(nofair_z, 2))
    print(alloc.to_string(index=False))


if __name__ == "__main__":
    main()
