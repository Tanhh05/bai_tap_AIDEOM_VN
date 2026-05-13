# AIDEOM-VN Practice Project

Project structure follows the assignment file `bai_tap_AIDEOM_VN.pdf`.

## Structure

```text
AIDEOM-VN/
├── data/                  # CSV data required by PDF
│   ├── vietnam_macro_2020_2025.csv
│   ├── vietnam_sectors_2024.csv
│   ├── vietnam_regions_2024.csv
│   └── vietnam_priorities.csv
├── notebooks/             # Jupyter notebooks by exercise
│   ├── ex01_cobb_douglas/
│   └── ex02_budget_lp/
├── src/
│   ├── utils/
│   ├── models/            # Exercise 1
│   ├── optimization/
│   ├── mcdm/
│   ├── rl/
│   └── dashboard/
├── results/               # charts and outputs
├── reports/               # exercise summaries and policy discussion
├── app.py                 # dashboard entrypoint
├── requirements.txt
└── README.md
```

## Setup

```bash
cd /Users/tanh/Desktop/bai_tap_AIDEOM_VN
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is intentionally lightweight for Streamlit Community Cloud. To run all optimization/RL scripts locally, install the full environment:

```bash
pip install -r requirements-dev.txt
```

Python 3.10+ is recommended by the PDF. If installing all full dependencies fails on Python 3.14, use Python 3.11 or 3.12.

## Run Completed Exercises

```bash
python3 src/models/ex1_cobb_douglas.py
python3 src/optimization/ex2_budget_lp.py
python3 src/mcdm/ex3_sector_priority.py
python3 src/optimization/ex4_regional_lp.py
python3 src/optimization/ex5_project_mip.py
python3 src/mcdm/ex6_topsis_regions.py
python3 src/optimization/ex7_pareto_nsga2.py
python3 src/optimization/ex8_dynamic_optimization.py
python3 src/optimization/ex9_labor_ai.py
python3 src/optimization/ex10_stochastic_programming.py
python3 src/rl/ex11_q_learning.py
python3 src/dashboard/ex12_integrated_aideom.py
python3 -m streamlit run app.py
```

## Notebooks

Each exercise has a runnable notebook under `notebooks/`. The notebooks call the corresponding Python script and inspect generated outputs in `results/`.

## Tests

```bash
pytest
```

The test suite covers the core numerical behavior of the main modules and verifies the known infeasibility of the hard Exercise 4 fairness constraint.

## Reports and Slides

- Full project report draft: `reports/aideom_vn_integrated_report.md`
- 15-slide presentation draft: `reports/aideom_vn_slides.md`

Or run all exercise scripts:

```bash
python3 scripts/run_all.py
```

Important modeling note: the hard regional fairness constraint in the PDF for Exercise 4 uses `lambda=0.70`, `gamma=0.002`, and a regional budget cap of `12000`. With the provided data this is infeasible for Tay Nguyen, so Exercise 4 reports the infeasibility and also solves a slack-penalized version. Exercise 7 keeps the feasible budget and human-capital constraints and records this fairness note in `results/ex7_model_notes.csv`.

## Execution order (from PDF)

1. Exercise 1: Cobb-Douglas + TFP decomposition
2. Exercise 2: Linear programming budget allocation
3. Exercise 3: Sector priority index
4. Exercise 4: Regional fair allocation LP
5. Exercise 5: MIP project selection
6. Exercise 6: TOPSIS (+ Entropy/AHP optional)
7. Exercise 7: Multi-objective optimization
8. Exercise 8: Dynamic intertemporal optimization
9. Exercise 9: Labor retraining optimization
10. Exercise 10: Two-stage stochastic programming
11. Exercise 11: Q-learning policy
12. Exercise 12: Integrated AIDEOM-VN prototype

## Current status

- [x] Project scaffold created
- [x] Exercise 1 implementation
- [x] Exercise 2 implementation
- [x] Exercise 3 implementation
- [x] Exercise 4 implementation
- [x] Exercise 5 implementation
- [x] Exercise 6 implementation
- [x] Exercise 7 implementation
- [x] Exercise 8 implementation
- [x] Exercise 9 implementation
- [x] Exercise 10 implementation
- [x] Exercise 11 implementation
- [x] Exercise 12 integration
