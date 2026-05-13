from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pulp
from scipy.optimize import linprog


ITEMS = [
    "Ha tang so",
    "AI va du lieu",
    "Nhan luc so",
    "R&D cong nghe",
]
COEFFICIENTS = [0.85, 1.20, 0.95, 1.35]


@dataclass(frozen=True)
class BudgetScenario:
    budget: float = 100.0
    min_infrastructure: float = 25.0
    min_ai: float = 15.0
    min_human_capital: float = 20.0
    min_rd: float = 10.0
    strategic_share: float = 0.35


def linprog_matrices(s: BudgetScenario) -> tuple[list[float], list[list[float]], list[float]]:
    c = [-v for v in COEFFICIENTS]
    a_ub = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [s.strategic_share, s.strategic_share - 1, s.strategic_share, s.strategic_share - 1],
    ]
    b_ub = [
        s.budget,
        -s.min_infrastructure,
        -s.min_ai,
        -s.min_human_capital,
        -s.min_rd,
        0,
    ]
    return c, a_ub, b_ub


def solve_with_scipy(s: BudgetScenario) -> dict[str, float | str]:
    c, a_ub, b_ub = linprog_matrices(s)
    res = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=[(0, None)] * 4, method="highs")
    if not res.success:
        return {"status": res.message, "Z": float("nan"), **{item: float("nan") for item in ITEMS}}

    allocation = {item: float(value) for item, value in zip(ITEMS, res.x)}
    return {"status": "optimal", "Z": float(-res.fun), **allocation}


def solve_with_pulp(s: BudgetScenario) -> tuple[dict[str, float | str], pd.DataFrame]:
    model = pulp.LpProblem("AIDEOM_VN_Exercise_2_Budget_Allocation", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ITEMS, lowBound=0)

    model += pulp.lpSum(COEFFICIENTS[i] * x[ITEMS[i]] for i in range(4)), "Expected_GDP_Gain"
    constraints = {
        "budget_total": pulp.lpSum(x[item] for item in ITEMS) <= s.budget,
        "min_infrastructure": x["Ha tang so"] >= s.min_infrastructure,
        "min_ai_data": x["AI va du lieu"] >= s.min_ai,
        "min_human_capital": x["Nhan luc so"] >= s.min_human_capital,
        "min_rd": x["R&D cong nghe"] >= s.min_rd,
        "strategic_technology_share": x["AI va du lieu"] + x["R&D cong nghe"]
        >= s.strategic_share * pulp.lpSum(x[item] for item in ITEMS),
    }
    for name, constraint in constraints.items():
        model += constraint, name

    model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[model.status].lower()
    allocation = {item: float(pulp.value(x[item])) for item in ITEMS}
    result = {"status": status, "Z": float(pulp.value(model.objective)), **allocation}

    duals = []
    for name, constraint in model.constraints.items():
        duals.append(
            {
                "constraint": name,
                "shadow_price": float(constraint.pi),
                "slack": float(constraint.slack),
            }
        )
    return result, pd.DataFrame(duals)


def sensitivity_by_budget(budgets: list[float]) -> pd.DataFrame:
    rows = []
    for budget in budgets:
        row = solve_with_scipy(BudgetScenario(budget=budget))
        row["budget"] = budget
        rows.append(row)
    return pd.DataFrame(rows)


def compare_human_capital_priority() -> pd.DataFrame:
    base = solve_with_scipy(BudgetScenario())
    priority = solve_with_scipy(BudgetScenario(min_human_capital=30.0))
    return pd.DataFrame(
        [
            {"scenario": "base_x3_min_20", **base},
            {"scenario": "priority_x3_min_30", **priority},
        ]
    )


def save_sensitivity_plot(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(df["budget"], df["Z"], marker="o")
    plt.title("Exercise 2 - Sensitivity of optimal GDP gain Z*(B)")
    plt.xlabel("Budget B (thousand billion VND)")
    plt.ylabel("Optimal expected GDP gain Z*")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = BudgetScenario()
    scipy_result = pd.DataFrame([solve_with_scipy(base)])
    pulp_result, duals = solve_with_pulp(base)
    pulp_result_df = pd.DataFrame([pulp_result])
    sensitivity = sensitivity_by_budget([100.0, 120.0, 140.0])
    human_capital = compare_human_capital_priority()

    scipy_result.to_csv(out_dir / "ex2_scipy_solution.csv", index=False)
    pulp_result_df.to_csv(out_dir / "ex2_pulp_solution.csv", index=False)
    duals.to_csv(out_dir / "ex2_pulp_duals.csv", index=False)
    sensitivity.to_csv(out_dir / "ex2_budget_sensitivity.csv", index=False)
    human_capital.to_csv(out_dir / "ex2_human_capital_priority.csv", index=False)
    save_sensitivity_plot(sensitivity, out_dir / "ex2_budget_sensitivity.png")

    print("=== Exercise 2 Completed ===")
    print("SciPy solution:")
    print(scipy_result.to_string(index=False))
    print("\nPuLP solution:")
    print(pulp_result_df.to_string(index=False))
    print("\nDual values:")
    print(duals.to_string(index=False))
    print("\nBudget sensitivity:")
    print(sensitivity[["budget", "Z", *ITEMS]].to_string(index=False))
    print("\nHuman capital priority scenario:")
    print(human_capital[["scenario", "status", "Z", *ITEMS]].to_string(index=False))


if __name__ == "__main__":
    main()
