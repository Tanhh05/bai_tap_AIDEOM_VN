from __future__ import annotations

from pathlib import Path

import pandas as pd
import pulp
import pyomo.environ as pyo


ITEMS = ["I", "D", "AI", "H"]
BASE = {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95}
PROB = {"s1": 0.30, "s2": 0.45, "s3": 0.20, "s4": 0.05}
BETA_S = {
    "s1": {"I": 1.25, "D": 1.35, "AI": 1.55, "H": 1.05},
    "s2": {"I": 1.00, "D": 1.10, "AI": 1.25, "H": 0.95},
    "s3": {"I": 0.75, "D": 0.85, "AI": 0.90, "H": 1.00},
    "s4": {"I": 0.40, "D": 0.50, "AI": 0.55, "H": 1.10},
}


def solve_sp(fixed_x: dict[str, float] | None = None, scenario: str | None = None) -> tuple[dict, pd.DataFrame]:
    m = pulp.LpProblem("VN_Two_Stage_SP", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ITEMS, lowBound=0)
    scenarios = [scenario] if scenario else list(PROB)
    prob = {scenario: 1.0} if scenario else PROB
    y = pulp.LpVariable.dicts("y", (scenarios, ITEMS), lowBound=0)
    beta_second = {s: BETA_S[s] for s in scenarios}
    m += pulp.lpSum(BASE[j] * x[j] for j in ITEMS) + pulp.lpSum(prob[s] * pulp.lpSum(beta_second[s][j] * y[s][j] for j in ITEMS) for s in scenarios)
    m += pulp.lpSum(x[j] for j in ITEMS) <= 65000
    for j, val in (fixed_x or {}).items():
        m += x[j] == val
    for s in scenarios:
        m += pulp.lpSum(y[s][j] for j in ITEMS) <= 15000
        m += y[s]["AI"] <= 0.5 * x["H"]
    m.solve(pulp.PULP_CBC_CMD(msg=False))
    summary = {"status": pulp.LpStatus[m.status].lower(), "Z": float(pulp.value(m.objective) or 0), **{f"x_{j}": float(pulp.value(x[j]) or 0) for j in ITEMS}}
    rows = []
    for s in scenarios:
        rows.append({"scenario": s, **{f"y_{j}": float(pulp.value(y[s][j]) or 0) for j in ITEMS}})
    return summary, pd.DataFrame(rows)


def solve_sp_pyomo() -> tuple[dict, pd.DataFrame]:
    """Pyomo implementation of the same two-stage model.

    GLPK/CBC command-line solvers are not always installed. PuLP ships a CBC
    binary, so Pyomo is pointed at that executable to keep the project runnable.
    """
    m = pyo.ConcreteModel()
    m.J = pyo.Set(initialize=ITEMS)
    m.S = pyo.Set(initialize=list(PROB))
    m.p = pyo.Param(m.S, initialize=PROB)
    m.beta = pyo.Param(m.J, initialize=BASE)
    m.beta_s = pyo.Param(m.S, m.J, initialize={(s, j): BETA_S[s][j] for s in PROB for j in ITEMS})
    m.x = pyo.Var(m.J, within=pyo.NonNegativeReals)
    m.y = pyo.Var(m.S, m.J, within=pyo.NonNegativeReals)
    m.budget1 = pyo.Constraint(expr=sum(m.x[j] for j in m.J) <= 65000)
    m.budget2 = pyo.Constraint(m.S, rule=lambda mod, s: sum(mod.y[s, j] for j in mod.J) <= 15000)
    m.ai_h_link = pyo.Constraint(m.S, rule=lambda mod, s: mod.y[s, "AI"] <= 0.5 * mod.x["H"])
    m.obj = pyo.Objective(
        expr=sum(m.beta[j] * m.x[j] for j in m.J)
        + sum(m.p[s] * sum(m.beta_s[s, j] * m.y[s, j] for j in m.J) for s in m.S),
        sense=pyo.maximize,
    )

    cbc_path = pulp.PULP_CBC_CMD().path
    solver = pyo.SolverFactory("cbc", executable=cbc_path)
    result = solver.solve(m)
    status = str(result.solver.termination_condition).lower()
    summary = {"status": status, "Z": float(pyo.value(m.obj)), **{f"x_{j}": float(pyo.value(m.x[j])) for j in ITEMS}}
    rows = []
    for s in PROB:
        rows.append({"scenario": s, **{f"y_{j}": float(pyo.value(m.y[s, j])) for j in ITEMS}})
    return summary, pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    sp_summary, sp_y = solve_sp()
    pyomo_summary, pyomo_y = solve_sp_pyomo()
    deterministic = []
    wait_and_see = 0.0
    for s, p in PROB.items():
        sol, _ = solve_sp(scenario=s)
        deterministic.append({"case": f"deterministic_{s}", **sol})
        wait_and_see += p * sol["Z"]
    avg_beta = {j: sum(PROB[s] * BETA_S[s][j] for s in PROB) for j in ITEMS}
    ev_item = max(avg_beta, key=avg_beta.get)
    fixed = {j: (65000.0 if j == ev_item else 0.0) for j in ITEMS}
    ev_eval, _ = solve_sp(fixed_x=fixed)
    summary = pd.DataFrame([{"case": "stochastic_pulp", **sp_summary}, {"case": "stochastic_pyomo", **pyomo_summary}, {"case": "expected_value_policy", **ev_eval}, *deterministic, {"case": "metrics", "status": "computed", "Z": sp_summary["Z"], "x_I": 0, "x_D": 0, "x_AI": wait_and_see - sp_summary["Z"], "x_H": sp_summary["Z"] - ev_eval["Z"]}])
    summary.rename(columns={"x_AI": "EVPI_or_x_AI", "x_H": "VSS_or_x_H"}, inplace=True)
    summary.to_csv(out / "ex10_stochastic_summary.csv", index=False)
    sp_y.to_csv(out / "ex10_stochastic_second_stage.csv", index=False)
    pyomo_y.to_csv(out / "ex10_pyomo_second_stage.csv", index=False)
    print("=== Exercise 10 Completed ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
