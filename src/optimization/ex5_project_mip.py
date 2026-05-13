from __future__ import annotations

from pathlib import Path

import pandas as pd
import pulp


PROJECTS = [
    (1, "Trung tam du lieu quoc gia Hoa Lac", "Ha tang", 12000, 21500, 8500, 3500),
    (2, "Trung tam du lieu quoc gia phia Nam", "Ha tang", 11500, 20800, 7500, 4000),
    (3, "He thong 5G phu song toan quoc", "Ha tang", 18000, 32500, 12000, 6000),
    (4, "He thong dinh danh dien tu VNeID 2.0", "Chinh phu so", 4500, 9200, 3500, 1000),
    (5, "Cong dich vu cong quoc gia v3", "Chinh phu so", 3200, 6800, 2500, 700),
    (6, "Y te so quoc gia", "Y te so", 5800, 11400, 4000, 1800),
    (7, "Giao duc so K-12 toan quoc", "Giao duc", 6500, 12200, 4500, 2000),
    (8, "Trung tam AI quoc gia + supercomputing", "AI", 15000, 28500, 9000, 6000),
    (9, "Sandbox tai chinh so", "Tai chinh so", 2500, 5800, 1800, 700),
    (10, "Logistics thong minh + cang bien so", "Logistics", 7200, 13800, 5000, 2200),
    (11, "Nong nghiep so DBSCL", "Nong nghiep", 4800, 8500, 3500, 1300),
    (12, "Dao tao 50000 ky su AI/ban dan", "Nhan luc", 8500, 16200, 5500, 3000),
    (13, "Khu CN ban dan Bac Ninh - Bac Giang", "Ban dan", 20000, 35000, 13000, 7000),
    (14, "An ninh mang quoc gia SOC", "An ninh", 3800, 7500, 2800, 1000),
    (15, "Open Data + du lieu mo quoc gia", "Du lieu", 1500, 3800, 1200, 300),
]


def project_df() -> pd.DataFrame:
    return pd.DataFrame(PROJECTS, columns=["id", "project_name", "sector", "cost", "benefit", "cost_year_1_2", "cost_year_3_5"])


def completion_probability(sector: str) -> float:
    if sector == "Ha tang":
        return 0.85
    if sector == "Chinh phu so":
        return 0.75
    if sector in {"AI", "Ban dan"}:
        return 0.65
    return 0.80


def solve_project_selection(budget: float = 80000, require_both_data_centers: bool = False, expected: bool = False) -> tuple[pd.DataFrame, dict[str, float | str]]:
    df = project_df()
    pids = df["id"].tolist()
    row = df.set_index("id")
    benefit = row["benefit"].to_dict()
    if expected:
        benefit = {i: benefit[i] * completion_probability(row.loc[i, "sector"]) for i in pids}

    m = pulp.LpProblem("VN_Project_Selection", pulp.LpMaximize)
    y = pulp.LpVariable.dicts("y", pids, cat="Binary")
    m += pulp.lpSum(benefit[i] * y[i] for i in pids)
    m += pulp.lpSum(row.loc[i, "cost"] * y[i] for i in pids) <= budget, "budget_total"
    m += pulp.lpSum(row.loc[i, "cost_year_1_2"] * y[i] for i in pids) <= 40000, "budget_year_1_2"
    if require_both_data_centers:
        m += y[1] == 1
        m += y[2] == 1
    else:
        m += y[1] + y[2] <= 1, "data_center_exclusive"
    m += y[8] <= y[12], "ai_requires_training"
    m += y[13] <= y[12], "semiconductor_requires_training"
    m += y[4] + y[5] >= 1, "digital_government_min"
    m += y[14] >= 1, "cybersecurity_required"
    m += pulp.lpSum(y[i] for i in pids) >= 7, "min_projects"
    m += pulp.lpSum(y[i] for i in pids) <= 11, "max_projects"
    m.solve(pulp.PULP_CBC_CMD(msg=False))

    selected_ids = [i for i in pids if pulp.value(y[i]) and pulp.value(y[i]) > 0.5]
    selected = df[df["id"].isin(selected_ids)].copy()
    z = float(pulp.value(m.objective)) if m.objective is not None else float("nan")
    total_cost = float(selected["cost"].sum())
    summary = {
        "status": pulp.LpStatus[m.status].lower(),
        "budget": budget,
        "expected_objective": expected,
        "require_both_data_centers": require_both_data_centers,
        "Z": z,
        "total_cost": total_cost,
        "total_year_1_2": float(selected["cost_year_1_2"].sum()),
        "project_count": len(selected),
        "benefit_cost_ratio": z / total_cost if total_cost else float("nan"),
    }
    return selected, summary


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    project_df().to_csv(out / "ex5_project_catalog.csv", index=False)

    scenarios = [
        ("base_budget_80000", solve_project_selection()),
        ("budget_100000", solve_project_selection(budget=100000)),
        ("require_p1_p2", solve_project_selection(require_both_data_centers=True)),
        ("expected_benefit", solve_project_selection(expected=True)),
    ]
    summaries = []
    for name, (selected, summary) in scenarios:
        selected.to_csv(out / f"ex5_{name}_selected.csv", index=False)
        summaries.append({"scenario": name, **summary, "selected_projects": ",".join(f"P{i}" for i in selected["id"])})
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "ex5_scenario_summary.csv", index=False)

    print("=== Exercise 5 Completed ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
