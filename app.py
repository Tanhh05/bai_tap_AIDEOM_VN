from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide")

root = Path(__file__).resolve().parent
results_dir = root / "results"

MENU_OPTIONS = {
    "Trang chủ": "Trang chủ",
    "Bài 1 - Cobb-Douglas + TFP": "Bài 1 - Cobb-Douglas + TFP",
    "Bài 2 - LP ngân sách": "Bài 2 - LP ngân sách số",
    "Bài 3 - Priority ngành": "Bài 3 - Priority 10 ngành",
    "Bài 4 - LP đa vùng": "Bài 4 - LP ngành-vùng",
    "Bài 5 - MIP dự án": "Bài 5 - MIP 15 dự án",
    "Bài 6 - TOPSIS": "Bài 6 - TOPSIS 6 vùng",
    "Bài 7 - Đa mục tiêu": "Bài 7 - NSGA-II Pareto",
    "Bài 8 - Tối ưu động": "Bài 8 - Động 2026-2035",
    "Bài 9 - Việc làm ròng": "Bài 9 - Lao động & AI",
    "Bài 10 - Stochastic Programming": "Bài 10 - Stochastic SP",
    "Bài 11 - Q-learning": "Bài 11 - Q-learning RL",
    "Bài 12 - Tích hợp hệ thống": "Bài 12 - AIDEOM tích hợp",
}


st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 75% 0%, #2f3a46 0, #1f2730 32%, #151a21 100%);
        color: #eef3f7;
    }
    .main-title {
        font-size: clamp(2rem, 4vw, 3.6rem);
        line-height: .95;
        font-weight: 900;
        margin: 0 0 .4rem 0;
        color: #f4f7fb;
    }
    .soft-caption {
        color: #d9e1e8;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.07);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 8px;
        padding: .85rem .9rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background: linear-gradient(90deg, #ffd7e3, #ff9fca);
        color: #25313b;
        border-radius: 8px 8px 0 0;
        font-weight: 800;
    }
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def csv_path(name: str) -> Path:
    return results_dir / name


def read_csv(name: str) -> pd.DataFrame | None:
    path = csv_path(name)
    if not path.exists():
        return None
    return pd.read_csv(path)


def missing_files(names: list[str]) -> list[str]:
    return [name for name in names if not csv_path(name).exists()]


def numeric_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    exclude = exclude or set()
    return [col for col in df.select_dtypes("number").columns if col not in exclude]


def chart_block(df: pd.DataFrame, title: str, index: str | None = None, columns: list[str] | None = None, kind: str = "bar") -> None:
    st.markdown(f"**{title}**")
    plot_df = df.copy()
    if index and index in plot_df.columns:
        plot_df = plot_df.set_index(index)
    if columns:
        columns = [col for col in columns if col in plot_df.columns]
        plot_df = plot_df[columns]
    else:
        cols = numeric_columns(plot_df)
        plot_df = plot_df[cols]
    if plot_df.empty:
        st.info("Không có cột số phù hợp để vẽ biểu đồ.")
        return
    if kind == "line":
        st.line_chart(plot_df)
    elif kind == "area":
        st.area_chart(plot_df)
    else:
        st.bar_chart(plot_df)


def image_grid(image_files: list[str]) -> None:
    existing = [csv_path(name) for name in image_files if csv_path(name).exists()]
    if not existing:
        return
    cols = st.columns(2)
    for i, img in enumerate(existing):
        with cols[i % 2]:
            st.image(str(img), caption=img.name, width="stretch")


def radar_chart(df: pd.DataFrame, title: str = "Radar 4 KPI cho 5 kịch bản") -> None:
    kpi_map = {
        "GDP_2030": "GDP gain",
        "D_2030": "Số hóa",
        "AI_2030": "AI",
        "H_2030": "Nhân lực",
    }
    required = ["scenario", *kpi_map.keys()]
    if any(col not in df.columns for col in required):
        st.info("Thiếu dữ liệu để vẽ radar KPI.")
        return

    data = df[required].copy().head(5)
    metrics = list(kpi_map.keys())
    mins = data[metrics].min()
    spans = (data[metrics].max() - mins).replace(0, 1)
    normalized = (data[metrics] - mins) / spans

    size = 560
    center = size / 2
    radius = 190
    colors = ["#ffb3c7", "#6ee7d8", "#9fb7ff", "#ffd166", "#b8f2a6"]

    def point(axis_idx: int, value: float) -> tuple[float, float]:
        angle = -math.pi / 2 + 2 * math.pi * axis_idx / len(metrics)
        return center + radius * value * math.cos(angle), center + radius * value * math.sin(angle)

    rings = []
    for level in range(1, 6):
        value = level / 5
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in [point(i, value) for i in range(len(metrics))])
        rings.append(f'<polygon points="{pts}" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="1"/>')

    axes = []
    labels = []
    for i, metric in enumerate(metrics):
        x, y = point(i, 1)
        axes.append(f'<line x1="{center}" y1="{center}" x2="{x:.1f}" y2="{y:.1f}" stroke="rgba(255,255,255,.30)" stroke-width="1"/>')
        lx, ly = point(i, 1.15)
        labels.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" dominant-baseline="middle" '
            f'fill="#f2f6fa" font-size="14" font-weight="700">{escape(kpi_map[metric])}</text>'
        )

    polygons = []
    legend = []
    for row_idx, (_, row) in enumerate(data.iterrows()):
        pts = " ".join(
            f"{x:.1f},{y:.1f}"
            for x, y in [point(i, float(normalized.iloc[row_idx][metric])) for i, metric in enumerate(metrics)]
        )
        color = colors[row_idx % len(colors)]
        polygons.append(
            f'<polygon points="{pts}" fill="{color}" fill-opacity=".18" stroke="{color}" stroke-width="2.5"/>'
        )
        polygons.extend(
            f'<circle cx="{point(i, float(normalized.iloc[row_idx][metric]))[0]:.1f}" '
            f'cy="{point(i, float(normalized.iloc[row_idx][metric]))[1]:.1f}" r="3.5" fill="{color}"/>'
            for i, metric in enumerate(metrics)
        )
        y = 82 + row_idx * 24
        legend.append(f'<rect x="405" y="{y}" width="12" height="12" rx="2" fill="{color}"/>')
        legend.append(
            f'<text x="424" y="{y + 10}" fill="#dce6ee" font-size="12">{escape(str(row["scenario"]))}</text>'
        )

    html = f"""
    <div style="width:100%;display:flex;justify-content:center;">
      <svg viewBox="0 0 {size} {size}" style="max-width:760px;width:100%;height:auto;">
        <rect width="{size}" height="{size}" rx="10" fill="rgba(255,255,255,.035)"/>
        <text x="28" y="38" fill="#f4f7fb" font-size="24" font-weight="800">{escape(title)}</text>
        <text x="28" y="63" fill="#cfd8df" font-size="13">So sánh đa chiều 4 KPI, chuẩn hóa min-max theo 5 kịch bản</text>
        {''.join(rings)}
        {''.join(axes)}
        {''.join(polygons)}
        <circle cx="{center}" cy="{center}" r="3" fill="#f4f7fb"/>
        {''.join(labels)}
        {''.join(legend)}
      </svg>
    </div>
    """
    components.html(html, height=610)


def data_tabs(csv_files: list[str]) -> None:
    existing = [(name, read_csv(name)) for name in csv_files if read_csv(name) is not None]
    if not existing:
        return
    tabs = st.tabs([name.replace(".csv", "") for name, _ in existing])
    for tab, (name, df) in zip(tabs, existing):
        with tab:
            st.dataframe(df, width="stretch")


def show_result_files(title: str, csv_files: list[str], image_files: list[str]) -> None:
    st.subheader(title)
    missing = missing_files([*csv_files, *image_files])
    if missing:
        st.warning("Một số file kết quả chưa có. Hãy chạy script tương ứng trước.")
        st.write("Thiếu:", missing)
    image_grid(image_files)
    data_tabs(csv_files)


def render_sidebar() -> str:
    with st.sidebar:
        selected = st.selectbox(
            "Chọn bài",
            list(MENU_OPTIONS.keys()),
            index=len(MENU_OPTIONS) - 1,
        )
    return MENU_OPTIONS[selected]


exercise = render_sidebar()

st.markdown('<div class="main-title">VN Bài 12 - AIDEOM-VN Dashboard tích hợp</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="soft-caption">Mô hình AIDEOM-VN tích hợp 6 module: Dự báo, ngân sách số, phân bổ vùng, lao động, rủi ro và chính sách.</div>',
    unsafe_allow_html=True,
)

top_tab, model_tab, policy_tab = st.tabs(["📍 Bài tập", "📊 Dashboard M1-M4", "⚠️ Nhận xét chính sách"])

with top_tab:
    if exercise == "Trang chủ":
        st.subheader("Tổng quan kết quả")
        scenario_df = read_csv("ex12_scenario_dashboard_summary.csv")
        if scenario_df is not None:
            best = scenario_df.sort_values("GDP_2030", ascending=False).iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Số bài", "12")
            c2.metric("Kịch bản tốt nhất", best["scenario"])
            c3.metric("GDP 2030", f"{best['GDP_2030']:,.0f}")
            c4.metric("Số kịch bản", len(scenario_df))
            left, right = st.columns(2)
            with left:
                chart_block(scenario_df, "GDP 2030 theo kịch bản", "scenario", ["GDP_2030"])
            with right:
                chart_block(scenario_df, "Cấu phần phân bổ K-D-AI-H", "scenario", ["K", "D", "AI", "H"])
        st.info("Chọn từng bài trong thanh menu bên trái để xem biểu đồ và bảng kết quả chi tiết.")

    elif exercise == "Bài 1 - Cobb-Douglas + TFP":
        st.subheader("Bài 1 - Cobb-Douglas + TFP")
        tfp_df = read_csv("ex1_tfp_and_prediction.csv")
        decomp_df = read_csv("ex1_growth_decomposition.csv")
        share_df = read_csv("ex1_growth_contribution_share.csv")
        if tfp_df is None or decomp_df is None or share_df is None:
            st.error("Thiếu file kết quả Bài 1. Hãy chạy: `python3 src/models/ex1_cobb_douglas.py`")
        else:
            y_2025 = float(tfp_df.loc[tfp_df["year"] == 2025, "GDP_trillion_VND"].iloc[0])
            yhat_2025 = float(tfp_df.loc[tfp_df["year"] == 2025, "Y_hat_from_A_bar"].iloc[0])
            mape = (abs((tfp_df["GDP_trillion_VND"] - tfp_df["Y_hat_from_A_bar"]) / tfp_df["GDP_trillion_VND"]).mean()) * 100
            tfp_growth = (float(tfp_df["A_t"].iloc[-1]) / float(tfp_df["A_t"].iloc[0]) - 1) * 100
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("GDP 2025", f"{y_2025:,.1f}")
            c2.metric("Y-hat 2025", f"{yhat_2025:,.1f}")
            c3.metric("MAPE", f"{mape:.2f}%")
            c4.metric("TFP tăng", f"{tfp_growth:.2f}%")
            image_grid(["ex1_tfp_trend.png", "ex1_growth_contribution_share.png"])
            left, right = st.columns(2)
            with left:
                chart_block(tfp_df, "GDP thực tế và dự báo", "year", ["GDP_trillion_VND", "Y_hat_from_A_bar"], "line")
                chart_block(tfp_df, "Các biến đầu vào", "year", ["K_trillion_VND", "L_million_workers", "D_digital_pct", "AI_thousand_firms", "H_trained_labor_pct"], "line")
            with right:
                chart_block(tfp_df, "TFP A_t", "year", ["A_t"], "line")
                chart_block(share_df, "Tỷ trọng đóng góp tăng trưởng", "component", ["share_pct"])
            data_tabs(["ex1_tfp_and_prediction.csv", "ex1_growth_decomposition.csv", "ex1_growth_contribution_share.csv"])

    elif exercise == "Bài 2 - LP ngân sách số":
        st.subheader("Bài 2 - Phân bổ ngân sách đầu tư số")
        scipy_df = read_csv("ex2_scipy_solution.csv")
        sensitivity_df = read_csv("ex2_budget_sensitivity.csv")
        if scipy_df is None or sensitivity_df is None:
            st.error("Thiếu file kết quả Bài 2. Hãy chạy: `python3 src/optimization/ex2_budget_lp.py`")
        else:
            best = scipy_df.iloc[0]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Z* cơ sở", f"{best['Z']:.2f}")
            c2.metric("Hạ tầng", f"{best['Ha tang so']:.1f}")
            c3.metric("AI + dữ liệu", f"{best['AI va du lieu']:.1f}")
            c4.metric("Nhân lực", f"{best['Nhan luc so']:.1f}")
            c5.metric("R&D", f"{best['R&D cong nghe']:.1f}")
            image_grid(["ex2_budget_sensitivity.png"])
            left, right = st.columns(2)
            with left:
                chart_block(scipy_df, "Cơ cấu nghiệm tối ưu", None, ["Ha tang so", "AI va du lieu", "Nhan luc so", "R&D cong nghe"])
                chart_block(sensitivity_df, "Z* theo ngân sách", "budget", ["Z"], "line")
            with right:
                chart_block(sensitivity_df, "Phân bổ theo các mức ngân sách", "budget", ["Ha tang so", "AI va du lieu", "Nhan luc so", "R&D cong nghe"], "area")
                dual_df = read_csv("ex2_pulp_duals.csv")
                if dual_df is not None:
                    chart_block(dual_df, "Shadow price theo ràng buộc", "constraint", ["shadow_price"])
            data_tabs(["ex2_scipy_solution.csv", "ex2_pulp_solution.csv", "ex2_pulp_duals.csv", "ex2_budget_sensitivity.csv", "ex2_human_capital_priority.csv"])

    elif exercise == "Bài 3 - Priority 10 ngành":
        st.subheader("Bài 3 - Chỉ số ưu tiên ngành")
        rank_df = read_csv("ex3_sector_priority_rank.csv")
        sens_df = read_csv("ex3_ai_weight_sensitivity.csv")
        image_grid(["ex3_ai_weight_sensitivity_heatmap.png"])
        if rank_df is not None:
            chart_block(rank_df, "Xếp hạng Priority theo ngành", "sector_name_vi", ["Priority"])
        if sens_df is not None:
            pivot = sens_df.pivot_table(index="ai_weight", columns="sector_name_vi", values="rank", aggfunc="min")
            st.markdown("**Độ nhạy thứ hạng theo trọng số AI**")
            st.line_chart(pivot)
        data_tabs(["ex3_sector_priority_rank.csv", "ex3_normalized_matrix.csv", "ex3_policy_weight_top3.csv", "ex3_ai_weight_sensitivity.csv"])

    elif exercise == "Bài 4 - LP ngành-vùng":
        st.subheader("Bài 4 - Phân bổ ngành-vùng")
        image_grid(["ex4_allocation_heatmap.png"])
        alloc_df = read_csv("ex4_pulp_allocation.csv")
        no_fair_df = read_csv("ex4_no_fairness_allocation.csv")
        comp_df = read_csv("ex4_objective_comparison.csv")
        left, right = st.columns(2)
        with left:
            if alloc_df is not None:
                chart_block(alloc_df, "Phân bổ có slack fairness", "region_name_vi", ["I", "D", "AI", "H"], "bar")
        with right:
            if no_fair_df is not None:
                chart_block(no_fair_df, "Phân bổ không fairness", "region_name_vi", ["I", "D", "AI", "H"], "bar")
        if comp_df is not None:
            chart_block(comp_df.fillna(0), "So sánh mục tiêu mô hình", "model", ["Z"])
        data_tabs(["ex4_objective_comparison.csv", "ex4_pulp_allocation.csv", "ex4_no_fairness_allocation.csv", "ex4_cvxpy_allocation.csv"])

    elif exercise == "Bài 5 - MIP 15 dự án":
        st.subheader("Bài 5 - Lựa chọn danh mục dự án")
        summary_df = read_csv("ex5_scenario_summary.csv")
        catalog_df = read_csv("ex5_project_catalog.csv")
        if summary_df is not None:
            c1, c2, c3 = st.columns(3)
            c1.metric("Kịch bản", len(summary_df))
            c2.metric("Z cao nhất", f"{summary_df['Z'].max():,.0f}")
            c3.metric("Chi phí thấp nhất", f"{summary_df['total_cost'].min():,.0f}")
            chart_block(summary_df, "Z và tổng chi phí theo kịch bản", "scenario", ["Z", "total_cost"])
            chart_block(summary_df, "Số dự án và B/C ratio", "scenario", ["project_count", "benefit_cost_ratio"])
        if catalog_df is not None:
            left, right = st.columns(2)
            with left:
                chart_block(catalog_df, "Chi phí theo dự án", "project_name", ["cost"])
            with right:
                chart_block(catalog_df, "Lợi ích theo dự án", "project_name", ["benefit"])
        data_tabs(["ex5_scenario_summary.csv", "ex5_project_catalog.csv", "ex5_base_budget_80000_selected.csv", "ex5_budget_100000_selected.csv", "ex5_require_p1_p2_selected.csv", "ex5_expected_benefit_selected.csv"])

    elif exercise == "Bài 6 - TOPSIS 6 vùng":
        st.subheader("Bài 6 - TOPSIS vùng")
        image_grid(["ex6_topsis_expert_scores.png"])
        expert_df = read_csv("ex6_topsis_expert_rank.csv")
        entropy_df = read_csv("ex6_topsis_entropy_rank.csv")
        weights_df = read_csv("ex6_entropy_weights.csv")
        left, right = st.columns(2)
        with left:
            if expert_df is not None:
                chart_block(expert_df, "TOPSIS điểm chuyên gia", "region_name_vi", ["TOPSIS_score"])
        with right:
            if entropy_df is not None:
                chart_block(entropy_df, "TOPSIS trọng số entropy", "region_name_vi", ["TOPSIS_score"])
        if weights_df is not None:
            chart_block(weights_df, "Trọng số entropy theo tiêu chí", "criterion", ["entropy_weight"])
        data_tabs(["ex6_topsis_expert_rank.csv", "ex6_topsis_entropy_rank.csv", "ex6_entropy_weights.csv", "ex6_ai_weight_sensitivity.csv"])

    elif exercise == "Bài 7 - NSGA-II Pareto":
        st.subheader("Bài 7 - Đa mục tiêu Pareto")
        image_grid(["ex7_pareto_3d.png", "ex7_parallel_coordinates.png"])
        pareto_df = read_csv("ex7_pareto_objectives.csv")
        alloc_df = read_csv("ex7_compromise_allocation.csv")
        if pareto_df is not None:
            chart_block(pareto_df.head(30), "30 nghiệm Pareto đầu tiên", None, ["growth", "inclusion_mad", "emission", "security_risk"], "line")
            chart_block(pareto_df, "Phân phối mục tiêu Pareto", None, ["growth", "emission", "security_risk"])
        if alloc_df is not None:
            chart_block(alloc_df, "Phân bổ nghiệm compromise", "region_code", ["I", "D", "AI", "H"])
        data_tabs(["ex7_pareto_objectives.csv", "ex7_compromise_allocation.csv", "ex7_opportunity_cost.csv", "ex7_model_notes.csv"])

    elif exercise == "Bài 8 - Động 2026-2035":
        st.subheader("Bài 8 - Tối ưu động")
        image_grid(["ex8_dynamic_trajectories.png"])
        opt_df = read_csv("ex8_dynamic_optimal_path.csv")
        shock_df = read_csv("ex8_dynamic_shock_path.csv")
        shares_df = read_csv("ex8_optimal_investment_shares.csv")
        if opt_df is not None:
            chart_block(opt_df, "Quỹ đạo Y, C và A", "year", ["Y", "C", "A"], "line")
            chart_block(opt_df, "Đầu tư theo cấu phần", "year", ["I_K", "I_D", "I_AI", "I_H"], "area")
        if shock_df is not None:
            chart_block(shock_df, "Kịch bản shock", "year", ["Y", "C", "A"], "line")
        if shares_df is not None:
            chart_block(shares_df, "Tỷ trọng đầu tư tối ưu", "year", ["share_K", "share_D", "share_AI", "share_H"], "area")
        data_tabs(["ex8_dynamic_optimal_path.csv", "ex8_dynamic_shock_path.csv", "ex8_optimal_investment_shares.csv", "ex8_strategy_comparison.csv"])

    elif exercise == "Bài 9 - Lao động & AI":
        st.subheader("Bài 9 - Lao động và AI")
        image_grid(["ex9_vulnerable_labor_flow.png"])
        labor_df = read_csv("ex9_labor_allocation.csv")
        limited_df = read_csv("ex9_labor_displacement_limited.csv")
        summary_df = read_csv("ex9_summary.csv")
        if labor_df is not None:
            chart_block(labor_df, "Việc làm theo ngành", "sector_name_vi", ["NewJob", "UpgradeJob", "DisplacedJob", "RetrainingCapacity", "NetJob"])
        if limited_df is not None:
            chart_block(limited_df, "Kịch bản giới hạn displaced", "sector_name_vi", ["NewJob", "UpgradeJob", "DisplacedJob", "NetJob"])
        if summary_df is not None:
            chart_block(summary_df, "Tổng việc làm ròng", "case", ["total_NetJob"])
        data_tabs(["ex9_labor_allocation.csv", "ex9_labor_displacement_limited.csv", "ex9_summary.csv"])

    elif exercise == "Bài 10 - Stochastic SP":
        st.subheader("Bài 10 - Stochastic Programming")
        summary_df = read_csv("ex10_stochastic_summary.csv")
        second_df = read_csv("ex10_stochastic_second_stage.csv")
        pyomo_df = read_csv("ex10_pyomo_second_stage.csv")
        if summary_df is not None:
            chart_block(summary_df, "First-stage quyết định và giá trị", "case", ["Z", "x_I", "x_D", "EVPI_or_x_AI", "VSS_or_x_H"])
        if second_df is not None:
            chart_block(second_df, "Second-stage PuLP", "scenario", ["y_I", "y_D", "y_AI", "y_H"])
        if pyomo_df is not None:
            chart_block(pyomo_df, "Second-stage Pyomo", "scenario", ["y_I", "y_D", "y_AI", "y_H"])
        data_tabs(["ex10_stochastic_summary.csv", "ex10_stochastic_second_stage.csv", "ex10_pyomo_second_stage.csv"])

    elif exercise == "Bài 11 - Q-learning RL":
        st.subheader("Bài 11 - Q-learning")
        image_grid(["ex11_learning_curve.png"])
        curve_df = read_csv("ex11_learning_curve.csv")
        compare_df = read_csv("ex11_policy_comparison.csv")
        if curve_df is not None:
            chart_block(curve_df, "Learning curve reward", "episode", ["reward"], "line")
            chart_block(curve_df.tail(200), "Reward 200 episode cuối", "episode", ["reward"], "line")
        if compare_df is not None:
            chart_block(compare_df, "So sánh chính sách", "policy", ["avg_reward"])
        data_tabs(["ex11_policy_samples.csv", "ex11_policy_comparison.csv", "ex11_learning_curve.csv"])

    elif exercise == "Bài 12 - AIDEOM tích hợp":
        st.subheader("Bài 12 - Nguyên mẫu AIDEOM-VN")
        scenario_df = read_csv("ex12_scenario_dashboard_summary.csv")
        if scenario_df is None:
            st.warning("Chưa có kết quả Bài 12. Hãy chạy: `python3 src/dashboard/ex12_integrated_aideom.py`")
        else:
            budget = st.slider("Tổng ngân sách hiện hành (tỷ VND)", 50000, 100000, 80000, 5000)
            st.caption(f"Ngân sách mô phỏng: {budget:,} tỷ VND")
            tab_overview, tab_alloc, tab_scenario, tab_risk = st.tabs(["Tổng quan", "Phân bổ", "Kịch bản so sánh", "Cảnh báo rủi ro"])
            with tab_overview:
                best = scenario_df.sort_values("GDP_2030", ascending=False).iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Kịch bản GDP 2030 cao nhất", best["scenario"])
                c2.metric("GDP 2030", f"{best['GDP_2030']:,.1f}")
                c3.metric("Số kịch bản", len(scenario_df))
                radar_chart(scenario_df)
                chart_block(scenario_df, "GDP 2030", "scenario", ["GDP_2030"])
            with tab_alloc:
                chart_block(scenario_df, "Tỷ trọng phân bổ chính sách", "scenario", ["K", "D", "AI", "H"], "area")
                chart_block(scenario_df, "Mức trạng thái 2030", "scenario", ["K_2030", "D_2030", "AI_2030", "H_2030"])
            with tab_scenario:
                radar_chart(scenario_df, "Radar KPI theo kịch bản chính sách")
                chart_block(scenario_df, "So sánh đầu ra 2030", "scenario", ["GDP_2030", "K_2030", "D_2030", "AI_2030", "H_2030"], "line")
                show_result_files("Module outputs M1-M5", ["ex1_tfp_and_prediction.csv", "ex4_objective_comparison.csv", "ex6_topsis_expert_rank.csv", "ex9_summary.csv", "ex10_stochastic_summary.csv"], [])
            with tab_risk:
                risk_df = scenario_df[["scenario", "risk_flags"]].copy()
                st.dataframe(risk_df, width="stretch")
                risky = risk_df[~risk_df["risk_flags"].eq("OK")]
                if risky.empty:
                    st.success("Không có cảnh báo rủi ro theo ngưỡng hiện tại.")
                else:
                    st.warning("Các kịch bản còn cảnh báo rủi ro cần diễn giải chính sách.")

with model_tab:
    st.subheader("Dashboard M1-M4")
    scenario_df = read_csv("ex12_scenario_dashboard_summary.csv")
    if scenario_df is not None:
        radar_chart(scenario_df)
        left, right = st.columns(2)
        with left:
            chart_block(scenario_df, "M1 - GDP dự báo 2030", "scenario", ["GDP_2030"])
            chart_block(scenario_df, "M2 - Phân bổ K-D-AI-H", "scenario", ["K", "D", "AI", "H"])
        with right:
            chart_block(scenario_df, "M3 - Trạng thái 2030", "scenario", ["K_2030", "D_2030", "AI_2030", "H_2030"])
            st.dataframe(scenario_df, width="stretch")
    else:
        st.warning("Chưa có dữ liệu tích hợp Bài 12.")

with policy_tab:
    st.subheader("Nhận xét chính sách")
    scenario_df = read_csv("ex12_scenario_dashboard_summary.csv")
    if scenario_df is not None:
        risky = scenario_df[~scenario_df["risk_flags"].eq("OK")]
        c1, c2, c3 = st.columns(3)
        c1.metric("Kịch bản", len(scenario_df))
        c2.metric("Có cảnh báo", len(risky))
        c3.metric("GDP 2030 TB", f"{scenario_df['GDP_2030'].mean():,.0f}")
        st.dataframe(scenario_df[["scenario", "GDP_2030", "risk_flags"]], width="stretch")
    st.info("Các biểu đồ dùng trực tiếp dữ liệu trong thư mục `results`, nên khi chạy lại script kết quả dashboard sẽ tự cập nhật.")
