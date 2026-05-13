from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "src/models/ex1_cobb_douglas.py",
    "src/optimization/ex2_budget_lp.py",
    "src/mcdm/ex3_sector_priority.py",
    "src/optimization/ex4_regional_lp.py",
    "src/optimization/ex5_project_mip.py",
    "src/mcdm/ex6_topsis_regions.py",
    "src/optimization/ex7_pareto_nsga2.py",
    "src/optimization/ex8_dynamic_optimization.py",
    "src/optimization/ex9_labor_ai.py",
    "src/optimization/ex10_stochastic_programming.py",
    "src/rl/ex11_q_learning.py",
    "src/dashboard/ex12_integrated_aideom.py",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for script in SCRIPTS:
        print(f"\n=== RUN {script} ===")
        subprocess.run([sys.executable, script], cwd=root, check=True)


if __name__ == "__main__":
    main()
