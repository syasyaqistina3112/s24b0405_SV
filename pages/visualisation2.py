import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="Malaysia Household Income vs. Inflation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Malaysia Household Income Growth vs. CPI Inflation (2018–2024)")
st.info(
    "Objective 2: To compare household income growth against consumer prices to identify periods where price increases outpace income."
)
st.markdown(
    "Analysis of state-level household income growth against Consumer Price Index (CPI) inflation "
    "using data from the Department of Statistics Malaysia (DOSM)."  
)

# 2. Load Data with Caching (Prevents re-downloading on every interaction)
@st.cache_data
def load_data():
    url_income = "https://storage.dosm.gov.my/hies/hh_income_state.parquet"
    url_cpi = "https://storage.dosm.gov.my/cpi/cpi_2d_state.parquet"

    df_income = pd.read_parquet(url_income)
    if "date" in df_income.columns:
        df_income["date"] = pd.to_datetime(df_income["date"])

    df_cpi = pd.read_parquet(url_cpi)
    if "date" in df_cpi.columns:
        df_cpi["date"] = pd.to_datetime(df_cpi["date"])

    return df_income, df_cpi

with st.spinner("Loading DOSM datasets..."):
    df_income, df_cpi = load_data()

# 3. Filter Data (2018 - 2024 inclusive)
df_income_filtered = df_income[
    (df_income["date"] >= "2018-01-01") & (df_income["date"] <= "2024-12-31")
]
df_cpi_filtered = df_cpi[
    (df_cpi["date"] >= "2018-01-01") & (df_cpi["date"] <= "2024-12-31")
]

# Keep only 'overall' division for general inflation comparison
df_cpi_overall = df_cpi_filtered[df_cpi_filtered["division"] == "overall"]

# Inner join on state and date
df_merged = pd.merge(
    df_income_filtered,
    df_cpi_overall,
    on=["state", "date"],
    how="inner"
).rename(columns={"index": "cpi_overall"})

# Sort by state and date for accurate period-over-period growth calculations
df_merged_sorted = df_merged.sort_values(by=["state", "date"]).copy()

# Calculate period-over-period growth rates (%) within each state
df_merged_sorted["income_mean_growth"] = (
    df_merged_sorted.groupby("state")["income_mean"].pct_change() * 100
)
df_merged_sorted["income_median_growth"] = (
    df_merged_sorted.groupby("state")["income_median"].pct_change() * 100
)
df_merged_sorted["cpi_growth"] = (
    df_merged_sorted.groupby("state")["cpi_overall"].pct_change() * 100
)

# Calculate the gap where inflation growth exceeds mean income growth
df_merged_sorted["inflation_gap_mean"] = (
    df_merged_sorted["cpi_growth"] - df_merged_sorted["income_mean_growth"]
)

# Drop the first period (no prior year for growth calculation)
df_growth = df_merged_sorted.dropna(subset=["income_mean_growth", "cpi_growth"]).copy()
df_growth["year"] = df_growth["date"].dt.year

# 4. Sidebar Filter (Defaults to All States, matching your original script)
st.sidebar.header("Filter Options")
state_options = ["All States (Average)"] + sorted(df_merged["state"].unique().tolist())
selected_state = st.sidebar.selectbox("Select State:", state_options)

if selected_state == "All States (Average)":
    df_growth_view = df_growth.copy()
    df_real_view = df_merged.copy()
else:
    df_growth_view = df_growth[df_growth["state"] == selected_state].copy()
    df_real_view = df_merged[df_merged["state"] == selected_state].copy()

# 5. Summary Metrics & KPI Cards
avg_income_growth = df_growth_view["income_mean_growth"].mean()
avg_cpi_growth = df_growth_view["cpi_growth"].mean()

yearly_gap = df_growth_view.groupby("year")["inflation_gap_mean"].mean().reset_index()
most_pronounced_year = yearly_gap.loc[yearly_gap["inflation_gap_mean"].idxmax()]

val_income = f"{avg_income_growth:.2f}%"
val_cpi = f"{avg_cpi_growth:.2f}%"
val_gap_year = str(int(most_pronounced_year["year"]))
val_gap_diff = f"{most_pronounced_year['inflation_gap_mean']:.2f}%"

html_cards = f"""
<div style="display: flex; gap: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 15px 0 25px 0;">
    <div style="flex: 1; background-color: #f8f9fa; border: 1px solid #e9ecef; border-left: 5px solid #28a745; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="font-size: 13px; color: #6c757d; font-weight: 600; text-transform: uppercase;">Avg Income Growth</div>
        <div style="font-size: 28px; font-weight: bold; color: #212529; margin: 5px 0;">{val_income}</div>
        <div style="font-size: 12px; color: #28a745;">▲ Household Mean</div>
    </div>
    <div style="flex: 1; background-color: #f8f9fa; border: 1px solid #e9ecef; border-left: 5px solid #dc3545; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="font-size: 13px; color: #6c757d; font-weight: 600; text-transform: uppercase;">Avg Price Increase</div>
        <div style="font-size: 28px; font-weight: bold; color: #212529; margin: 5px 0;">{val_cpi}</div>
        <div style="font-size: 12px; color: #dc3545;">▲ Consumer Price Index</div>
    </div>
    <div style="flex: 1; background-color: #f8f9fa; border: 1px solid #e9ecef; border-left: 5px solid #fd7e14; padding: 15px; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <div style="font-size: 13px; color: #6c757d; font-weight: 600; text-transform: uppercase;">Max Inflation Gap Year</div>
        <div style="font-size: 28px; font-weight: bold; color: #212529; margin: 5px 0;">{val_gap_year}</div>
        <div style="font-size: 12px; color: #fd7e14;">▲ {val_gap_diff} Inflation Surplus</div>
    </div>
</div>
"""
st.markdown(html_cards, unsafe_allow_html=True)

# 6. Visualizations
sns.set_theme(style="whitegrid")

# --- Chart 1: Grouped Bar Chart ---
st.subheader("1. Household Income Growth vs. CPI Inflation")
yearly_comparison = (
    df_growth_view.groupby("year")[["income_mean_growth", "cpi_growth"]]
    .mean()
    .reset_index()
)

df_melted = yearly_comparison.melt(
    id_vars="year",
    value_vars=["income_mean_growth", "cpi_growth"],
    var_name="Metric",
    value_name="Percentage Growth"
)

metric_map = {
    "income_mean_growth": "Average Income Growth (%)",
    "cpi_growth": "Average CPI Growth (%)"
}
df_melted["Metric"] = df_melted["Metric"].map(metric_map)

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(
    x="year",
    y="Percentage Growth",
    hue="Metric",
    data=df_melted,
    palette=["#28a745", "#dc3545"],
    ax=ax1
)

ax1.set_title(f"Average Household Income Growth vs. CPI Inflation ({selected_state})", fontsize=13, pad=15)
ax1.set_xlabel("Year", fontsize=11)
ax1.set_ylabel("Growth Rate (%)", fontsize=11)
ax1.legend(title="Metric", loc="upper right")

for p in ax1.patches:
    height = p.get_height()
    if not pd.isna(height) and height != 0:
        va_dir = "bottom" if height >= 0 else "top"
        xy_offset = (0, 6) if height >= 0 else (0, -12)
        ax1.annotate(
            f"{height:.2f}%",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va=va_dir,
            xytext=xy_offset,
            textcoords="offset points",
            fontsize=9,
            fontweight="bold"
        )

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)

# --- Chart 2 & Chart 3 Side by Side ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("2. Net Gap (Income Growth minus Inflation)")
    st.caption("Red bars indicate periods where inflation grew faster than average household income.")

    yearly_comparison["net_surplus"] = (
        yearly_comparison["income_mean_growth"] - yearly_comparison["cpi_growth"]
    )
    colors = ["#dc3545" if x < 0 else "#28a745" for x in yearly_comparison["net_surplus"]]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x="year",
        y="net_surplus",
        hue="year",
        data=yearly_comparison,
        palette=colors,
        legend=False,
        ax=ax2
    )

    ax2.set_title("Net Surplus / Deficit (%)", fontsize=12, pad=15)
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Net Surplus / Deficit (%)", fontsize=11)
    ax2.axhline(0, color="black", linewidth=1, linestyle="--")

    for p in ax2.patches:
        height = p.get_height()
        if not pd.isna(height):
            va_dir = "bottom" if height >= 0 else "top"
            xy_offset = (0, 5) if height >= 0 else (0, -14)
            ax2.annotate(
                f"{height:.2f}%",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va=va_dir,
                xytext=xy_offset,
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

with col2:
    st.subheader("3. Nominal vs. Real Household Income")
    st.caption("Real income is adjusted for inflation using 2019 as the base year (2019 = 100).")

    cpi_2019 = df_real_view[df_real_view["date"].dt.year == 2019]["cpi_overall"].mean()
    df_real_view["real_income_mean"] = (
        df_real_view["income_mean"] * (cpi_2019 / df_real_view["cpi_overall"])
    )

    yearly_trends = (
        df_real_view.groupby(df_real_view["date"].dt.year)[["income_mean", "real_income_mean"]]
        .mean()
        .reset_index()
    )

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(
        yearly_trends["date"],
        yearly_trends["income_mean"],
        marker="o",
        linewidth=2.5,
        color="#007bff",
        label="Nominal Income (Mean)"
    )
    ax3.plot(
        yearly_trends["date"],
        yearly_trends["real_income_mean"],
        marker="s",
        linewidth=2.5,
        color="#fd7e14",
        linestyle="--",
        label="Real Income (2019 Adjusted)"
    )

    for _, row in yearly_trends.iterrows():
        ax3.annotate(
            f"RM {row['income_mean']:.0f}",
            (row["date"], row["income_mean"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#0056b3"
        )
        ax3.annotate(
            f"RM {row['real_income_mean']:.0f}",
            (row["date"], row["real_income_mean"]),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color="#bd5b00"
        )

    ax3.set_title("Nominal vs. Real Average Household Income", fontsize=12, pad=15)
    ax3.set_xlabel("Year", fontsize=11)
    ax3.set_ylabel("Monthly Income (RM)", fontsize=11)
    ax3.set_xticks(yearly_trends["date"])
    ax3.legend(fontsize=10)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# 7. Optional Data Tables (Replaces Colab's display())
with st.expander("📋 View Processed Growth Data Table"):
    intervals_outpaced = df_growth_view[df_growth_view["inflation_gap_mean"] > 0]
    st.write(
        f"**Total intervals calculated:** {len(df_growth_view)} | "
        f"**Intervals where inflation outpaced mean income growth:** {len(intervals_outpaced)}"
    )
    st.dataframe(
        df_growth_view[
            ["state", "date", "income_mean", "income_mean_growth", "cpi_overall", "cpi_growth", "inflation_gap_mean"]
        ],
        use_container_width=True
    )
