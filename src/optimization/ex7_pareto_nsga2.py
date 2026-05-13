from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


class VietnamDigitalProblem(ElementwiseProblem):
    def __init__(self, regions: pd.DataFrame):
        super().__init__(n_var=24, n_obj=4, n_ieq_constr=14, xl=np.zeros(24), xu=np.ones(24) * 12000)
        self.beta = regions[["impact_I", "impact_D", "impact_AI", "impact_H"]].to_numpy(float)
        self.d0 = regions["digital_index_0_100"].to_numpy(float)
        self.e = regions["emission_coef"].to_numpy(float)
        self.rho = regions["risk_ai_coef"].to_numpy(float)
        self.sig = regions["risk_reduction_h_coef"].to_numpy(float)

    def _evaluate(self, x, out, *args, **kwargs):
        X = x.reshape(6, 4)
        region_budget = X.sum(axis=1)
        f_growth = -(self.beta * X).sum()
        f_inclusion = np.abs(region_budget - region_budget.mean()).mean()
        f_emission = (self.e * (X[:, 0] + X[:, 2])).sum()
        f_security = (self.rho * X[:, 2]).sum() - (self.sig * X[:, 3]).sum()
        g = [X.sum() - 50000, 12000 - X[:, 3].sum()]
        g += list(5000 - region_budget)
        g += list(region_budget - 12000)
        out["F"] = [f_growth, f_inclusion, f_emission, f_security]
        out["G"] = np.array(g)


def topsis_compromise(F: np.ndarray) -> int:
    scores = F.copy()
    scores[:, 0] = -scores[:, 0]
    beneficial = np.array([True, False, False, False])
    weights = np.array([0.40, 0.25, 0.20, 0.15])
    norm = (scores - scores.min(axis=0)) / (scores.max(axis=0) - scores.min(axis=0) + 1e-9)
    weighted = norm * weights
    ideal = np.where(beneficial, weighted.max(axis=0), weighted.min(axis=0))
    anti = np.where(beneficial, weighted.min(axis=0), weighted.max(axis=0))
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - anti) ** 2).sum(axis=1))
    return int(np.argmax(d_neg / (d_pos + d_neg + 1e-9)))


def save_plots(pareto: pd.DataFrame, out: Path) -> None:
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pareto["growth"], pareto["inclusion_mad"], pareto["emission"], s=18, alpha=0.75)
    ax.set_xlabel("Growth")
    ax.set_ylabel("Inclusion MAD")
    ax.set_zlabel("Emission")
    plt.tight_layout()
    plt.savefig(out / "ex7_pareto_3d.png", dpi=150)
    plt.close()

    cols = ["growth", "inclusion_mad", "emission", "security_risk"]
    scaled = (pareto[cols] - pareto[cols].min()) / (pareto[cols].max() - pareto[cols].min() + 1e-9)
    plt.figure(figsize=(8, 4))
    for _, row in scaled.sample(min(80, len(scaled)), random_state=42).iterrows():
        plt.plot(cols, row[cols], alpha=0.25, color="steelblue")
    plt.title("Exercise 7 - Parallel coordinates of Pareto objectives")
    plt.tight_layout()
    plt.savefig(out / "ex7_parallel_coordinates.png", dpi=150)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    regions = pd.read_csv(root / "data" / "vietnam_regions_2024.csv")
    res = minimize(VietnamDigitalProblem(regions), NSGA2(pop_size=100), ("n_gen", 200), seed=42, verbose=False)
    F = res.F
    X = res.X
    if F is None or X is None:
        raise RuntimeError("NSGA-II did not find feasible Pareto points. Check fairness constraints.")
    idx = topsis_compromise(F)
    pareto = pd.DataFrame({"growth": -F[:, 0], "inclusion_mad": F[:, 1], "emission": F[:, 2], "security_risk": F[:, 3]})
    pareto["is_compromise"] = False
    pareto.loc[idx, "is_compromise"] = True
    pareto.to_csv(out / "ex7_pareto_objectives.csv", index=False)
    pd.DataFrame(X[idx].reshape(6, 4), columns=["I", "D", "AI", "H"]).assign(region_code=regions["region_code"]).to_csv(out / "ex7_compromise_allocation.csv", index=False)
    pd.DataFrame([{"parameter": "fairness_C5", "value": "omitted", "note": "PDF lambda=0.70 is infeasible under regional budget cap 12000 as shown in Exercise 4; Exercise 7 keeps C1-C4 and C6 for feasible NSGA-II search."}]).to_csv(out / "ex7_model_notes.csv", index=False)
    save_plots(pareto, out)
    max_growth = pareto.loc[pareto["growth"].idxmax()]
    comp = pareto.loc[idx]
    pd.DataFrame([{
        "compromise_growth": comp["growth"],
        "max_growth": max_growth["growth"],
        "inclusion_penalty_pct": (max_growth["inclusion_mad"] / comp["inclusion_mad"] - 1) * 100 if comp["inclusion_mad"] else np.nan,
        "emission_penalty_pct": (max_growth["emission"] / comp["emission"] - 1) * 100 if comp["emission"] else np.nan,
    }]).to_csv(out / "ex7_opportunity_cost.csv", index=False)
    print("=== Exercise 7 Completed ===")
    print(pareto.loc[idx].to_string())


if __name__ == "__main__":
    main()
