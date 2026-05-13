from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide")
root = Path(__file__).resolve().parent
results_dir = root / "results"

st.title("AIDEOM-VN Dashboard")
st.caption("Integrated dashboard for Exercises 1-12")

exercise = st.sidebar.selectbox(
    "Chọn bài",
    [
        "Bài 1 - Cobb-Douglas + TFP",
        "Bài 2 - LP ngân sách",
        "Bài 3 - Priority ngành",
        "Bài 4 - LP đa vùng",
        "Bài 5 - MIP dự án",
        "Bài 6 - TOPSIS",
        "Bài 7 - Đa mục tiêu",
        "Bài 8 - Tối ưu động",
        "Bài 9 - Việc làm ròng",
        "Bài 10 - Stochastic Programming",
        "Bài 11 - Q-learning",
        "Bài 12 - Tích hợp hệ thống",
    ],
)


def show_result_files(title: str, csv_files: list[str], image_files: list[str]) -> None:
    st.subheader(title)
    existing_csv = [results_dir / f for f in csv_files if (results_dir / f).exists()]
    existing_img = [results_dir / f for f in image_files if (results_dir / f).exists()]
    missing = [f for f in [*csv_files, *image_files] if not (results_dir / f).exists()]
    if missing:
        st.warning("Một số file kết quả chưa có. Hãy chạy script tương ứng trước.")
        st.write("Thiếu:", missing)
    for img in existing_img:
        st.image(str(img), caption=img.name, width="stretch")
    for csv in existing_csv:
        st.markdown(f"**{csv.name}**")
        st.dataframe(pd.read_csv(csv), width="stretch")

if exercise == "Bài 1 - Cobb-Douglas + TFP":
    st.subheader("Bài 1 - Kết quả thực thi")

    tfp_file = results_dir / "ex1_tfp_and_prediction.csv"
    decomp_file = results_dir / "ex1_growth_decomposition.csv"
    share_file = results_dir / "ex1_growth_contribution_share.csv"
    tfp_plot = results_dir / "ex1_tfp_trend.png"
    share_plot = results_dir / "ex1_growth_contribution_share.png"

    missing = [p.name for p in [tfp_file, decomp_file, share_file, tfp_plot, share_plot] if not p.exists()]
    if missing:
        st.error("Thiếu file kết quả Bài 1. Hãy chạy: `python3 src/models/ex1_cobb_douglas.py`")
        st.write("Thiếu:", missing)
    else:
        tfp_df = pd.read_csv(tfp_file)
        share_df = pd.read_csv(share_file)
        decomp_df = pd.read_csv(decomp_file)

        y_2025 = float(tfp_df.loc[tfp_df["year"] == 2025, "GDP_trillion_VND"].iloc[0])
        yhat_2025 = float(tfp_df.loc[tfp_df["year"] == 2025, "Y_hat_from_A_bar"].iloc[0])
        mape = (abs((tfp_df["GDP_trillion_VND"] - tfp_df["Y_hat_from_A_bar"]) / tfp_df["GDP_trillion_VND"]).mean()) * 100
        tfp_growth = (float(tfp_df["A_t"].iloc[-1]) / float(tfp_df["A_t"].iloc[0]) - 1) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("GDP 2025", f"{y_2025:,.1f}")
        c2.metric("Ŷ 2025 (Ā)", f"{yhat_2025:,.1f}")
        c3.metric("MAPE", f"{mape:.2f}%")
        c4.metric("TFP tăng 2020→2025", f"{tfp_growth:.2f}%")

        col_left, col_right = st.columns(2)
        with col_left:
            st.image(str(tfp_plot), caption="Xu hướng A_t 2020-2025", width="stretch")
        with col_right:
            st.image(str(share_plot), caption="Tỷ trọng đóng góp tăng trưởng", width="stretch")

        st.markdown("**Bảng A_t và dự báo Ŷ**")
        st.dataframe(tfp_df, width="stretch")
        st.markdown("**Bảng phân rã tăng trưởng (log-diff)**")
        st.dataframe(decomp_df, width="stretch")
        st.markdown("**Tỷ trọng đóng góp (%)**")
        st.dataframe(share_df, width="stretch")
elif exercise == "Bài 2 - LP ngân sách":
    st.subheader("Bài 2 - Phân bổ ngân sách đầu tư số")

    scipy_file = results_dir / "ex2_scipy_solution.csv"
    pulp_file = results_dir / "ex2_pulp_solution.csv"
    dual_file = results_dir / "ex2_pulp_duals.csv"
    sensitivity_file = results_dir / "ex2_budget_sensitivity.csv"
    priority_file = results_dir / "ex2_human_capital_priority.csv"
    sensitivity_plot = results_dir / "ex2_budget_sensitivity.png"

    required = [scipy_file, pulp_file, dual_file, sensitivity_file, priority_file, sensitivity_plot]
    missing = [p.name for p in required if not p.exists()]
    if missing:
        st.error("Thiếu file kết quả Bài 2. Hãy chạy: `python3 src/optimization/ex2_budget_lp.py`")
        st.write("Thiếu:", missing)
    else:
        scipy_df = pd.read_csv(scipy_file)
        pulp_df = pd.read_csv(pulp_file)
        dual_df = pd.read_csv(dual_file)
        sensitivity_df = pd.read_csv(sensitivity_file)
        priority_df = pd.read_csv(priority_file)

        best = scipy_df.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Z* cơ sở", f"{best['Z']:.2f}")
        c2.metric("Hạ tầng số", f"{best['Ha tang so']:.1f}")
        c3.metric("AI + dữ liệu", f"{best['AI va du lieu']:.1f}")
        c4.metric("Nhân lực số", f"{best['Nhan luc so']:.1f}")
        c5.metric("R&D", f"{best['R&D cong nghe']:.1f}")

        st.markdown("**Nghiệm tối ưu bằng SciPy**")
        st.dataframe(scipy_df, width="stretch")
        st.markdown("**Nghiệm tối ưu bằng PuLP**")
        st.dataframe(pulp_df, width="stretch")
        st.markdown("**Giá đối ngẫu PuLP**")
        st.dataframe(dual_df, width="stretch")
        st.info(
            "Shadow price của ràng buộc ngân sách tổng là 1.35: trong vùng nghiệm hiện tại, "
            "tăng thêm 1 nghìn tỷ VND ngân sách làm Z* tăng khoảng 1.35 nghìn tỷ VND."
        )

        left, right = st.columns([1, 1])
        with left:
            st.image(str(sensitivity_plot), caption="Độ nhạy Z*(B)", width="stretch")
        with right:
            st.markdown("**Độ nhạy ngân sách**")
            st.dataframe(sensitivity_df, width="stretch")
            st.markdown("**Kịch bản x3 >= 30**")
            st.dataframe(priority_df, width="stretch")
elif exercise == "Bài 3 - Priority ngành":
    show_result_files(
        exercise,
        ["ex3_sector_priority_rank.csv", "ex3_normalized_matrix.csv", "ex3_policy_weight_top3.csv", "ex3_ai_weight_sensitivity.csv"],
        ["ex3_ai_weight_sensitivity_heatmap.png"],
    )
elif exercise == "Bài 4 - LP đa vùng":
    show_result_files(
        exercise,
        ["ex4_objective_comparison.csv", "ex4_pulp_allocation.csv", "ex4_no_fairness_allocation.csv", "ex4_cvxpy_allocation.csv"],
        ["ex4_allocation_heatmap.png"],
    )
elif exercise == "Bài 5 - MIP dự án":
    show_result_files(
        exercise,
        ["ex5_scenario_summary.csv", "ex5_project_catalog.csv", "ex5_base_budget_80000_selected.csv", "ex5_budget_100000_selected.csv", "ex5_require_p1_p2_selected.csv", "ex5_expected_benefit_selected.csv"],
        [],
    )
elif exercise == "Bài 6 - TOPSIS":
    show_result_files(
        exercise,
        ["ex6_topsis_expert_rank.csv", "ex6_topsis_entropy_rank.csv", "ex6_entropy_weights.csv", "ex6_ai_weight_sensitivity.csv"],
        ["ex6_topsis_expert_scores.png"],
    )
elif exercise == "Bài 7 - Đa mục tiêu":
    show_result_files(
        exercise,
        ["ex7_pareto_objectives.csv", "ex7_compromise_allocation.csv", "ex7_opportunity_cost.csv", "ex7_model_notes.csv"],
        ["ex7_pareto_3d.png", "ex7_parallel_coordinates.png"],
    )
elif exercise == "Bài 8 - Tối ưu động":
    show_result_files(
        exercise,
        ["ex8_dynamic_optimal_path.csv", "ex8_dynamic_shock_path.csv", "ex8_optimal_investment_shares.csv", "ex8_strategy_comparison.csv"],
        ["ex8_dynamic_trajectories.png"],
    )
elif exercise == "Bài 9 - Việc làm ròng":
    show_result_files(
        exercise,
        ["ex9_labor_allocation.csv", "ex9_labor_displacement_limited.csv", "ex9_summary.csv"],
        ["ex9_vulnerable_labor_flow.png"],
    )
elif exercise == "Bài 10 - Stochastic Programming":
    show_result_files(
        exercise,
        ["ex10_stochastic_summary.csv", "ex10_stochastic_second_stage.csv"],
        [],
    )
elif exercise == "Bài 11 - Q-learning":
    show_result_files(
        exercise,
        ["ex11_policy_samples.csv", "ex11_policy_comparison.csv", "ex11_learning_curve.csv"],
        ["ex11_learning_curve.png"],
    )
elif exercise == "Bài 12 - Tích hợp hệ thống":
    st.subheader("Bài 12 - Nguyên mẫu AIDEOM-VN")
    scenario_file = results_dir / "ex12_scenario_dashboard_summary.csv"
    if not scenario_file.exists():
        st.warning("Chưa có kết quả Bài 12. Hãy chạy: `python3 src/dashboard/ex12_integrated_aideom.py`")
    else:
        scenario_df = pd.read_csv(scenario_file)
        tab_overview, tab_alloc, tab_scenario, tab_risk = st.tabs(
            ["Tổng quan", "Phân bổ", "Kịch bản so sánh", "Cảnh báo rủi ro"]
        )
        with tab_overview:
            best = scenario_df.sort_values("GDP_2030", ascending=False).iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Kịch bản GDP 2030 cao nhất", best["scenario"])
            c2.metric("GDP 2030", f"{best['GDP_2030']:,.1f}")
            c3.metric("Số kịch bản", len(scenario_df))
            st.dataframe(scenario_df, width="stretch")
        with tab_alloc:
            st.bar_chart(scenario_df.set_index("scenario")[["K", "D", "AI", "H"]])
        with tab_scenario:
            st.line_chart(scenario_df.set_index("scenario")[["GDP_2030", "K_2030", "D_2030", "AI_2030", "H_2030"]])
            show_result_files(
                "Module outputs M1-M5",
                ["ex1_tfp_and_prediction.csv", "ex4_objective_comparison.csv", "ex6_topsis_expert_rank.csv", "ex9_summary.csv", "ex10_stochastic_summary.csv"],
                [],
            )
        with tab_risk:
            risk_df = scenario_df[["scenario", "risk_flags"]].copy()
            st.dataframe(risk_df, width="stretch")
            risky = risk_df[~risk_df["risk_flags"].eq("OK")]
            if risky.empty:
                st.success("Không có cảnh báo rủi ro theo ngưỡng hiện tại.")
            else:
                st.warning("Các kịch bản còn cảnh báo rủi ro cần diễn giải chính sách.")
else:
    st.subheader(exercise)
    st.info("Module này đang được triển khai theo thứ tự bài tập. Hiện đã hoàn tất phần chạy cho Bài 1.")

st.divider()
st.caption(f"Project root: {root}")
