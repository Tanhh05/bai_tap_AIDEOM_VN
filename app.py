from pathlib import Path
from html import escape
import math

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="AIDEOM-VN Dashboard", layout="wide")

root = Path(__file__).resolve().parent
results_dir = root / "results"

MENU_OPTIONS = [
    "Trang chủ",
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
]


st.markdown(
    """
    <style>
    html, body, [class*="css"], .stApp, .stMarkdown, .stDataFrame, .stSelectbox,
    .stButton, .stMetric, input, textarea, select, button {
        font-family: "Times New Roman", Times, serif !important;
    }
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


def normalized_score(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series([0.0] * len(df), index=df.index)
    values = df[available].astype(float)
    spans = (values.max() - values.min()).replace(0, 1)
    return ((values - values.min()) / spans).mean(axis=1)


def scenario_policy_text(scenario_df: pd.DataFrame) -> str:
    scored = scenario_df.copy()
    scored["balanced_score"] = normalized_score(scored, ["GDP_2030", "D_2030", "AI_2030", "H_2030"])
    best_gdp = scored.loc[scored["GDP_2030"].idxmax()]
    best_balanced = scored.loc[scored["balanced_score"].idxmax()]
    lowest_gdp = scored.loc[scored["GDP_2030"].idxmin()]
    avg_gdp = scored["GDP_2030"].mean()
    gdp_gap = best_gdp["GDP_2030"] - lowest_gdp["GDP_2030"]
    risky_count = int((~scored["risk_flags"].eq("OK")).sum())

    return f"""
**Phân tích tổng hợp.** Bộ 5 kịch bản cho thấy GDP 2030 trung bình đạt khoảng **{avg_gdp:,.0f}**,
trong đó kịch bản có GDP cao nhất là **{best_gdp['scenario']}** với **{best_gdp['GDP_2030']:,.0f}**.
Khoảng cách giữa kịch bản cao nhất và thấp nhất là **{gdp_gap:,.0f}**, nghĩa là lựa chọn chính sách
có ảnh hưởng đáng kể nhưng không chỉ nên nhìn vào GDP. Khi chuẩn hóa đồng thời 4 KPI
GDP, số hóa, AI và nhân lực, kịch bản cân bằng nhất là **{best_balanced['scenario']}**.

**Nhận xét chính sách.** Nếu ưu tiên tăng trưởng ngắn hạn, kịch bản GDP cao nhất là lựa chọn dễ bảo vệ
về mặt sản lượng. Tuy nhiên dashboard đang cho thấy **{risky_count}/{len(scored)}** kịch bản vẫn còn cảnh báo,
chủ yếu xoay quanh mục tiêu chuyển đổi số, nút thắt kỹ năng AI hoặc mức số hóa chưa đủ sâu. Vì vậy hướng triển khai
nên kết hợp đầu tư hạ tầng số với đào tạo nhân lực và năng lực hấp thụ AI, thay vì đẩy một cấu phần lên quá cao.

**Hàm ý triển khai.** Một kế hoạch khả thi nên lấy kịch bản cân bằng làm trục chính, sau đó tăng ngân sách có điều kiện
cho các cấu phần đang tạo biên lợi ích cao. Với các kịch bản bị cảnh báo rủi ro, cần gắn thêm chỉ tiêu trung gian:
tỷ lệ lao động được đào tạo lại, năng lực dữ liệu khu vực công, mức sẵn sàng AI theo ngành và khả năng bao trùm vùng.
Điều này giúp dashboard không chỉ là nơi xem kết quả mà còn là công cụ ra quyết định để kiểm tra đánh đổi giữa tăng trưởng,
chuyển đổi số và ổn định xã hội.
"""


def render_exercise_analysis(exercise: str) -> None:
    if exercise == "Trang chủ":
        return

    with st.expander("Phân tích và nhận xét chi tiết", expanded=True):
        if exercise == "Bài 1 - Cobb-Douglas + TFP":
            tfp_df = read_csv("ex1_tfp_and_prediction.csv")
            share_df = read_csv("ex1_growth_contribution_share.csv")
            if tfp_df is not None and share_df is not None:
                tfp_growth = (tfp_df["A_t"].iloc[-1] / tfp_df["A_t"].iloc[0] - 1) * 100
                mape = (abs((tfp_df["GDP_trillion_VND"] - tfp_df["Y_hat_from_A_bar"]) / tfp_df["GDP_trillion_VND"]).mean()) * 100
                top = share_df.sort_values("share_pct", ascending=False).iloc[0]
                st.markdown(
                    f"""
Mô hình Cobb-Douglas cho thấy TFP tăng khoảng **{tfp_growth:.2f}%** trong giai đoạn 2020-2025,
trong khi sai số MAPE của đường dự báo quanh **{mape:.2f}%**. Cấu phần đóng góp lớn nhất trong phân rã tăng trưởng
là **{top['component']}** với tỷ trọng **{top['share_pct']:.2f}%**. Điều này hàm ý tăng trưởng không chỉ phụ thuộc vào
vốn truyền thống mà còn phụ thuộc mạnh vào chất lượng công nghệ, mức số hóa và năng lực nhân lực.

Về chính sách, Bài 1 nên được dùng như lớp dự báo nền cho các bài sau. Nếu TFP tăng chậm, các kịch bản ngân sách
ở Bài 2 và phân bổ vùng ở Bài 4 cần ưu tiên những khoản đầu tư có khả năng lan tỏa năng suất thay vì chỉ mở rộng quy mô vốn.
"""
                )

        elif exercise == "Bài 2 - LP ngân sách":
            solution = read_csv("ex2_scipy_solution.csv")
            duals = read_csv("ex2_pulp_duals.csv")
            if solution is not None:
                row = solution.iloc[0]
                st.markdown(
                    f"""
Nghiệm tối ưu cơ sở đạt **Z = {row['Z']:.2f}**, với phân bổ lần lượt cho hạ tầng số, AI-dữ liệu,
nhân lực số và R&D là **{row['Ha tang so']:.1f}**, **{row['AI va du lieu']:.1f}**,
**{row['Nhan luc so']:.1f}** và **{row['R&D cong nghe']:.1f}**. Cấu trúc này cho thấy mô hình đang ưu tiên
R&D sau khi các ngưỡng tối thiểu ở các cấu phần nền tảng được đáp ứng.
"""
                )
            if duals is not None:
                active = duals.loc[duals["slack"].abs() < 1e-9, "constraint"].tolist()
                st.markdown(
                    f"""
Các ràng buộc đang chặt gồm: **{', '.join(active) if active else 'không có ràng buộc chặt rõ ràng'}**.
Khi ràng buộc ngân sách có shadow price dương, tăng ngân sách chỉ thực sự hiệu quả nếu vẫn nằm trong vùng nghiệm hiện tại.
Nếu muốn mở rộng phân tích, nên thêm kịch bản ngân sách theo vùng hoặc thêm ràng buộc tối thiểu cho đào tạo nhân lực.
"""
                )

        elif exercise == "Bài 3 - Priority ngành":
            rank_df = read_csv("ex3_sector_priority_rank.csv")
            if rank_df is not None:
                top3 = rank_df.head(3)
                st.markdown(
                    f"""
Ba ngành ưu tiên cao nhất là **{top3.iloc[0]['sector_name_vi']}**, **{top3.iloc[1]['sector_name_vi']}**
và **{top3.iloc[2]['sector_name_vi']}**. Chỉ số priority phản ánh đồng thời tăng trưởng, năng suất,
lan tỏa, xuất khẩu, lao động, sẵn sàng AI và rủi ro tự động hóa.

Khi trọng số AI thay đổi, thứ hạng ngành có thể dịch chuyển. Vì vậy kết quả Bài 3 không nên được hiểu là một bảng xếp hạng cố định,
mà là công cụ kiểm tra độ nhạy chính sách. Những ngành giữ thứ hạng cao qua nhiều trọng số là ứng viên tốt cho đầu tư dài hạn.
"""
                )

        elif exercise == "Bài 4 - LP đa vùng":
            comp_df = read_csv("ex4_objective_comparison.csv")
            alloc_df = read_csv("ex4_pulp_allocation.csv")
            if comp_df is not None:
                st.markdown(
                    """
Bài 4 cho thấy ràng buộc fairness cứng trong cấu hình dữ liệu hiện tại có thể làm mô hình không khả thi.
Đây là điểm quan trọng về mặt chính sách: công bằng vùng không thể chỉ đặt bằng một ngưỡng cứng nếu năng lực hấp thụ,
quy mô ngân sách vùng và điều kiện ban đầu quá khác nhau.
"""
                )
            if alloc_df is not None:
                top_region = alloc_df.sort_values("total", ascending=False).iloc[0]
                st.markdown(
                    f"""
Trong nghiệm có slack fairness, vùng nhận tổng phân bổ lớn nhất là **{top_region['region_name_vi']}**
với **{top_region['total']:,.0f}**. Cách đọc phù hợp là xem slack như chi phí chính sách của mục tiêu công bằng:
vùng nào cần slack lớn hơn thì cần bổ sung năng lực hấp thụ trước khi tăng ngân sách đầu tư.
"""
                )

        elif exercise == "Bài 5 - MIP dự án":
            summary_df = read_csv("ex5_scenario_summary.csv")
            if summary_df is not None:
                best = summary_df.sort_values("Z", ascending=False).iloc[0]
                efficient = summary_df.sort_values("benefit_cost_ratio", ascending=False).iloc[0]
                st.markdown(
                    f"""
Kịch bản có giá trị mục tiêu cao nhất là **{best['scenario']}** với **Z = {best['Z']:,.0f}**,
tổng chi phí **{best['total_cost']:,.0f}** và **{int(best['project_count'])}** dự án được chọn.
Kịch bản có tỷ lệ lợi ích/chi phí tốt nhất là **{efficient['scenario']}**.

MIP giúp tránh lựa chọn dự án theo cảm tính vì nó xét đồng thời ngân sách, lợi ích, chi phí giai đoạn đầu
và các điều kiện bắt buộc. Khi mở rộng, nên thêm ràng buộc về vùng, tiến độ giải ngân và phụ thuộc kỹ thuật giữa dự án.
"""
                )

        elif exercise == "Bài 6 - TOPSIS":
            expert_df = read_csv("ex6_topsis_expert_rank.csv")
            entropy_df = read_csv("ex6_topsis_entropy_rank.csv")
            if expert_df is not None and entropy_df is not None:
                st.markdown(
                    f"""
Theo trọng số chuyên gia, vùng đứng đầu là **{expert_df.iloc[0]['region_name_vi']}** với điểm TOPSIS
**{expert_df.iloc[0]['TOPSIS_score']:.3f}**. Theo trọng số entropy, vùng đứng đầu là
**{entropy_df.iloc[0]['region_name_vi']}** với điểm **{entropy_df.iloc[0]['TOPSIS_score']:.3f}**.

Sự khác biệt giữa hai cách gán trọng số cho thấy quyết định vùng ưu tiên phụ thuộc vào giả định chính sách.
Nếu kết quả chuyên gia và entropy cùng chỉ ra một vùng, đó là tín hiệu mạnh. Nếu khác nhau, cần thảo luận thêm về mục tiêu:
tăng trưởng, thu hút FDI, sẵn sàng số hay giảm chênh lệch vùng.
"""
                )

        elif exercise == "Bài 7 - Đa mục tiêu":
            opp_df = read_csv("ex7_opportunity_cost.csv")
            if opp_df is not None:
                row = opp_df.iloc[0]
                st.markdown(
                    f"""
Nghiệm compromise đạt tăng trưởng **{row['compromise_growth']:,.0f}** so với cực đại tăng trưởng
**{row['max_growth']:,.0f}**. Đổi lại, mô hình kiểm soát tốt hơn các mục tiêu phụ như bao trùm, phát thải
và rủi ro an ninh.

Bài 7 là phần thể hiện rõ nhất đánh đổi chính sách. Một nghiệm có GDP cao nhất chưa chắc là nghiệm tốt nhất nếu gây lệch vùng,
tăng phát thải hoặc làm rủi ro an ninh số cao hơn. Vì vậy nên dùng nghiệm compromise làm phương án trình bày chính.
"""
                )

        elif exercise == "Bài 8 - Tối ưu động":
            strategy_df = read_csv("ex8_strategy_comparison.csv")
            if strategy_df is not None:
                best = strategy_df.sort_values("welfare", ascending=False).iloc[0]
                st.markdown(
                    f"""
Chiến lược có phúc lợi cao nhất là **{best['strategy']}**, đạt welfare **{best['welfare']:.3f}**
và Y_2035 khoảng **{best['Y_2035']:,.0f}**. Quỹ đạo động cho thấy phân bổ đầu tư không chỉ quyết định GDP cuối kỳ
mà còn ảnh hưởng tiêu dùng và tích lũy trong toàn bộ giai đoạn.

Hàm ý là không nên dồn toàn bộ ngân sách vào một năm hoặc một cấu phần. Chính sách tốt cần nhịp đầu tư ổn định,
cho phép AI và hạ tầng số tạo tác động tích lũy trong nhiều năm.
"""
                )

        elif exercise == "Bài 9 - Việc làm ròng":
            summary_df = read_csv("ex9_summary.csv")
            if summary_df is not None:
                best = summary_df.sort_values("total_NetJob", ascending=False).iloc[0]
                st.markdown(
                    f"""
Kịch bản tốt nhất theo việc làm ròng là **{best['case']}**, tạo **{best['total_NetJob']:,.0f}**
việc làm ròng. Kết quả này cần được đọc cùng với rủi ro displaced job và năng lực đào tạo lại.

Nếu đầu tư AI tăng nhanh nhưng đào tạo lại không theo kịp, số việc làm mới có thể không bù được số lao động bị dịch chuyển.
Do đó chính sách AI cần đi kèm quỹ retraining, chuẩn kỹ năng số và cơ chế hỗ trợ nhóm lao động dễ tổn thương.
"""
                )

        elif exercise == "Bài 10 - Stochastic Programming":
            summary_df = read_csv("ex10_stochastic_summary.csv")
            if summary_df is not None:
                best = summary_df.sort_values("Z", ascending=False).iloc[0]
                st.markdown(
                    f"""
Mô hình stochastic đạt giá trị mục tiêu cao nhất **{best['Z']:,.0f}** trong trường hợp **{best['case']}**.
Điểm mạnh của Bài 10 là đưa bất định vào quyết định, thay vì tối ưu theo một kịch bản chắc chắn duy nhất.

Khi triển khai thực tế, first-stage decision nên là các khoản đầu tư khó đảo ngược như hạ tầng dữ liệu,
còn second-stage decision nên dành cho khoản linh hoạt như đào tạo, hỗ trợ chuyển đổi hoặc tăng cường năng lực vùng.
"""
                )

        elif exercise == "Bài 11 - Q-learning":
            compare_df = read_csv("ex11_policy_comparison.csv")
            if compare_df is not None:
                best = compare_df.sort_values("avg_reward", ascending=False).iloc[0]
                st.markdown(
                    f"""
Chính sách có reward trung bình cao nhất là **{best['policy']}** với **{best['avg_reward']:.3f}**.
Q-learning phù hợp để mô phỏng lựa chọn chính sách lặp lại, trong đó trạng thái nền kinh tế thay đổi sau mỗi quyết định.

Điểm cần lưu ý là RL không thay thế phân tích kinh tế, mà bổ sung một lớp học chính sách trong môi trường giả lập.
Khi mở rộng, nên định nghĩa reward đa mục tiêu hơn: GDP, bao trùm, rủi ro lao động, phát thải và an toàn dữ liệu.
"""
                )

        elif exercise == "Bài 12 - Tích hợp hệ thống":
            scenario_df = read_csv("ex12_scenario_dashboard_summary.csv")
            if scenario_df is not None:
                st.markdown(scenario_policy_text(scenario_df))


def render_sidebar() -> str:
    with st.sidebar:
        selected = st.selectbox(
            "Chọn bài",
            MENU_OPTIONS,
            index=0,
            key="exercise_selector",
        )
        st.caption(f"Đang xem: {selected}")
    return selected


exercise = render_sidebar()

page_title = "AIDEOM-VN Dashboard" if exercise == "Trang chủ" else exercise
st.markdown(f'<div class="main-title">{page_title}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="soft-caption">Mô hình AIDEOM-VN tích hợp 6 module: Dự báo, ngân sách số, phân bổ vùng, lao động, rủi ro và chính sách.</div>',
    unsafe_allow_html=True,
)

if exercise == "Bài 12 - Tích hợp hệ thống":
    top_tab, model_tab, policy_tab = st.tabs(["Bài tập", "Dashboard M1-M4", "Nhận xét chính sách"])
else:
    top_tab = st.container()
    model_tab = None
    policy_tab = None

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

    elif exercise == "Bài 2 - LP ngân sách":
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

    elif exercise == "Bài 3 - Priority ngành":
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

    elif exercise == "Bài 4 - LP đa vùng":
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

    elif exercise == "Bài 5 - MIP dự án":
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

    elif exercise == "Bài 6 - TOPSIS":
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

    elif exercise == "Bài 7 - Đa mục tiêu":
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

    elif exercise == "Bài 8 - Tối ưu động":
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

    elif exercise == "Bài 9 - Việc làm ròng":
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

    elif exercise == "Bài 10 - Stochastic Programming":
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

    elif exercise == "Bài 11 - Q-learning":
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

    elif exercise == "Bài 12 - Tích hợp hệ thống":
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

    render_exercise_analysis(exercise)

if model_tab is not None:
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

if policy_tab is not None:
    with policy_tab:
        st.subheader("Nhận xét chính sách")
        scenario_df = read_csv("ex12_scenario_dashboard_summary.csv")
        if scenario_df is not None:
            scored = scenario_df.copy()
            scored["balanced_score"] = normalized_score(scored, ["GDP_2030", "D_2030", "AI_2030", "H_2030"])
            risky = scenario_df[~scenario_df["risk_flags"].eq("OK")]
            c1, c2, c3 = st.columns(3)
            c1.metric("Kịch bản", len(scenario_df))
            c2.metric("Có cảnh báo", len(risky))
            c3.metric("GDP 2030 TB", f"{scenario_df['GDP_2030'].mean():,.0f}")
            st.markdown(scenario_policy_text(scenario_df))
            left, right = st.columns(2)
            with left:
                chart_block(scored, "Điểm cân bằng chuẩn hóa", "scenario", ["balanced_score"])
            with right:
                chart_block(scenario_df, "So sánh GDP và năng lực số", "scenario", ["GDP_2030", "D_2030", "AI_2030", "H_2030"], "line")
            st.markdown(
                """
**Khuyến nghị mở rộng phân tích.** Dashboard hiện có thể dùng như bản mẫu AIDEOM-VN để đọc nhanh từng module,
nhưng khi làm báo cáo dài hơn nên bổ sung ba lớp phân tích. Lớp thứ nhất là phân tích độ nhạy: thay đổi ngân sách,
trọng số AI, ngưỡng fairness và năng lực đào tạo để xem kết quả đảo chiều ở đâu. Lớp thứ hai là phân tích rủi ro:
gắn mỗi cảnh báo với một biến đo được, ví dụ tỷ lệ chuyển đổi số tối thiểu, số lao động được retraining, hay mức sẵn sàng AI.
Lớp thứ ba là phân tích thực thi: chia chính sách thành ngắn hạn, trung hạn và dài hạn để tránh kết luận chỉ dựa trên một năm 2030.

**Cách đọc kết quả.** Nếu một kịch bản có GDP cao nhưng điểm cân bằng thấp, đó là kịch bản thiên về tăng trưởng và cần kiểm tra
rủi ro xã hội. Nếu một kịch bản có GDP trung bình nhưng điểm cân bằng cao, đó có thể là phương án ổn định hơn để trình bày
trong bối cảnh chính sách công. Với những kịch bản có nhiều cảnh báo, không nên loại bỏ ngay; thay vào đó cần xem cảnh báo là
danh sách điều kiện đi kèm trước khi triển khai.
"""
            )
            st.dataframe(scenario_df[["scenario", "GDP_2030", "risk_flags"]], width="stretch")
        else:
            st.warning("Chưa có dữ liệu để tạo nhận xét chính sách.")
        st.info("Các biểu đồ dùng trực tiếp dữ liệu trong thư mục `results`, nên khi chạy lại script kết quả dashboard sẽ tự cập nhật.")
