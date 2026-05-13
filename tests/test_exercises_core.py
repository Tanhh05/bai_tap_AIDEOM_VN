from pathlib import Path

import numpy as np
import pandas as pd

from src.mcdm.ex6_topsis_regions import EXPERT_WEIGHTS, topsis
from src.models.ex1_cobb_douglas import Elasticities, estimate_tfp, load_macro_data
from src.optimization.ex2_budget_lp import BudgetScenario, solve_with_scipy
from src.optimization.ex4_regional_lp import solve_pulp
from src.optimization.ex5_project_mip import solve_project_selection
from src.optimization.ex9_labor_ai import solve_labor


ROOT = Path(__file__).resolve().parents[1]


def test_ex1_tfp_is_positive_and_growing():
    df = load_macro_data(ROOT / "data" / "vietnam_macro_2020_2025.csv")
    a = estimate_tfp(df, Elasticities())
    assert np.all(a > 0)
    assert a[-1] > a[0]


def test_ex2_budget_lp_base_solution():
    result = solve_with_scipy(BudgetScenario())
    assert result["status"] == "optimal"
    assert abs(result["Z"] - 112.25) < 1e-6
    assert result["R&D cong nghe"] == 40.0


def test_ex4_hard_fairness_is_infeasible_with_pdf_parameters():
    regions = pd.read_csv(ROOT / "data" / "vietnam_regions_2024.csv")
    _, _, status = solve_pulp(regions, fairness=True)
    assert status == "infeasible"


def test_ex5_project_selection_respects_constraints():
    selected, summary = solve_project_selection()
    ids = set(selected["id"])
    assert summary["status"] == "optimal"
    assert 7 <= len(ids) <= 11
    assert 14 in ids
    assert not ({1, 2} <= ids)


def test_ex6_topsis_ranks_southeast_first_with_expert_weights():
    regions = pd.read_csv(ROOT / "data" / "vietnam_regions_2024.csv")
    ranked = topsis(regions, EXPERT_WEIGHTS)
    assert ranked.iloc[0]["region_code"] == "SE"


def test_ex9_labor_solution_nonnegative_net_jobs():
    df, status, total = solve_labor()
    assert status == "optimal"
    assert total > 0
    assert (df["NetJob"] >= -1e-6).all()
