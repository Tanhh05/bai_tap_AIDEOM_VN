from __future__ import annotations

from pathlib import Path

import pandas as pd


SCENARIOS = {
    "S1_Truyen_thong": {"K": 0.70, "D": 0.10, "AI": 0.10, "H": 0.10},
    "S2_So_hoa_nhanh": {"K": 0.25, "D": 0.45, "AI": 0.15, "H": 0.15},
    "S3_AI_dan_dat": {"K": 0.20, "D": 0.20, "AI": 0.45, "H": 0.15},
    "S4_Bao_trum_so": {"K": 0.30, "D": 0.20, "AI": 0.10, "H": 0.40},
    "S5_Toi_uu_can_bang": {"K": 0.40, "D": 0.25, "AI": 0.15, "H": 0.20},
}


def project_macro_2030(weights: dict[str, float]) -> dict[str, float]:
    K, L, D, AI, H, A = 25900.0, 53.4, 19.5, 80.1, 29.2, 34.913621
    for _ in range(5):
        budget = 1000.0
        K *= 1.04 + 0.02 * weights["K"]
        L *= 1.006
        D += 2.0 * weights["D"]
        AI += 8.0 * weights["AI"]
        H += 1.5 * weights["H"]
        A *= 1 + 0.010 + 0.004 * weights["D"] + 0.003 * weights["AI"] + 0.003 * weights["H"]
    Y = A * K**0.33 * L**0.42 * D**0.10 * AI**0.08 * H**0.07
    return {"GDP_2030": Y, "K_2030": K, "D_2030": D, "AI_2030": AI, "H_2030": H}


def risk_flags(row: pd.Series) -> list[str]:
    flags = []
    if row["D_2030"] < 25:
        flags.append("Digital target risk")
    if row["AI_2030"] > 92 and row["H_2030"] < 31:
        flags.append("AI skill bottleneck")
    if row["scenario"].startswith("S1"):
        flags.append("Low digital transformation")
    return flags or ["OK"]


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    rows = []
    for scenario, weights in SCENARIOS.items():
        rows.append({"scenario": scenario, **weights, **project_macro_2030(weights)})
    df = pd.DataFrame(rows)
    df["risk_flags"] = df.apply(lambda r: "; ".join(risk_flags(r)), axis=1)
    df.to_csv(out / "ex12_scenario_dashboard_summary.csv", index=False)
    report = root / "reports" / "aideom_vn_integrated_report.md"
    report.parent.mkdir(exist_ok=True)
    report.write_text(
        "# AIDEOM-VN integrated prototype\n\n"
        "This report summarizes the runnable prototype modules M1-M6 mapped from Exercises 1-11.\n\n"
        + "```text\n"
        + df.to_string(index=False)
        + "\n```"
        + "\n",
        encoding="utf-8",
    )
    print("=== Exercise 12 Completed ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
