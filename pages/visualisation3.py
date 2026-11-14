import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Malaysia CPI, Income & Marriage Rate Analysis",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Impact of Cost of Living & Income on Marriage Rates in Malaysia (2018–2024)")
st.markdown(
    "Analyzing state-level Consumer Price Index (CPI), household income, and marriage rate trends "
    "using open data from the **Department of Statistics Malaysia (DOSM)**."
)

# ---------------------------------------------------------
# Data Loading & Preprocessing (Cached)
# ---------------------------------------------------------
@st.cache_data(ttl=86400)
def load_and_prepare_data():
    url_cpi = "https://storage.dosm.gov.my/cpi/cpi_2d_state.parquet"
    url_mar = "https://storage.dosm.gov.my/demography/marriages_state.parquet"
    url_income = "https://storage.dosm.gov.my/hies/hh_income_state.parquet"

    df_cpi = pd.read_parquet(url_cpi)
    df_mar = pd.read_parquet(url_mar)
    df_income = pd.read_parquet(url_income)

    # Ensure datetime conversion and extract year
    for df in (df_cpi, df_mar, df_income):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year

    # Filter for 2018 to 2024 inclusive
    df_cpi = df_cpi[(df_cpi["year"] >= 2018) & (df_cpi["year"] <= 2024)].copy()
    df_mar = df_mar[(df_mar["year"] >= 2018) & (df_mar["year"] <= 2024)].copy()
    df_income = df_income[(df_income["year"] >= 2018) & (df_income["year"] <= 2024)].copy()

    # Aggregate monthly CPI to annual averages per state and division
    df_cpi_annual = (
        df_cpi.groupby(["state", "year", "division"], as_index=False)["index"]
        .mean()
    )

    # Filter for 'overall' CPI division and compute true Year-over-Year (YoY) growth
    df_cpi_overall = (
        df_cpi_annual[df_cpi_annual["division"] == "overall"]
        .drop(columns=["division"])
        .sort_values(["state", "year"])
        .copy()
    )
    df_cpi_overall["cpi_growth_yoy"] = (
        df_cpi_overall.groupby("state")["index"].pct_change() * 100
    )

    # Merge Marriages with Overall CPI (2018-2024)
    df_mar_cpi = pd.merge(
        df_mar,
        df_cpi_overall,
        on=["state", "year"],
        how="inner"
    ).dropna(subset=["rate", "index"])

    # Merge with Household Income (available for HIES survey years, e.g., 2019 & 2022)
    df_full = pd.merge(
        df_mar_cpi,
        df_income[["state", "year", "income_mean", "income_median"]],
        on=["state", "year"],
        how="inner"
    )

    return df_mar_cpi, df_full


with st.spinner("Loading datasets from DOSM..."):
    df_mar_cpi, df_full = load_and_prepare_data()

# ---------------------------------------------------------
# Sidebar Filters
# ---------------------------------------------------------
st.sidebar.header("Filter Data")

available_states = sorted(df_mar_cpi["state"].unique())
selected_states = st.sidebar.multiselect(
    "Select States",
    options=available_states,
    default=available_states
)

available_sexes = sorted(df_mar_cpi["sex"].unique())
selected_sexes = st.sidebar.multiselect(
    "Select Sex",
    options=available_sexes,
    default=available_sexes
)

min_year, max_year = int(df_mar_cpi["year"].min()), int(df_mar_cpi["year"].max())
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Apply filters
df_filtered = df_mar_cpi[
    (df_mar_cpi["state"].isin(selected_states)) &
    (df_mar_cpi["sex"].isin(selected_sexes)) &
    (df_mar_cpi["year"].between(selected_years[0], selected_years[1]))
].copy()

df_full_filtered = df_full[
    (df_full["state"].isin(selected_states)) &
    (df_full["sex"].isin(selected_sexes)) &
    (df_full["year"].between(selected_years[0], selected_years[1]))
].copy()

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust your sidebar selection.")
    st.stop()

# ---------------------------------------------------------
# Summary Metrics (KPI Cards)
# ---------------------------------------------------------
st.subheader("Key Summary Metrics")

correlation = df_filtered["index"].corr(df_filtered["rate"])

# Identify peak inflation year and the lowest marriage rate in that year
peak_inflation_row = df_filtered.loc[df_filtered["index"].idxmax()]
peak_inflation_year = int(peak_inflation_row["year"])
peak_inflation_val = peak_inflation_row["index"]

df_peak_year = df_filtered[df_filtered["year"] == peak_inflation_year]
lowest_rate_row = df_peak_year.loc[df_peak_year["rate"].idxmin()]
lowest_rate_val = lowest_rate_row["rate"]

# Helper text for correlation strength
if pd.isna(correlation):
    corr_desc = "Insufficient variance to compute correlation"
elif abs(correlation) < 0.3:
    corr_desc = f"Weak {'negative' if correlation < 0 else 'positive'} relationship"
elif abs(correlation) < 0.7:
    corr_desc = f"Moderate {'negative' if correlation < 0 else 'positive'} relationship"
else:
    corr_desc = f"Strong {'negative' if correlation < 0 else 'positive'} relationship"

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Correlation (r): CPI vs. Marriage Rate",
        value=f"{correlation:.4f}" if not pd.isna(correlation) else "N/A",
        delta=corr_desc,
        delta_color="normal" if (not pd.isna(correlation) and correlation >= 0) else "inverse"
    )

with col2:
    st.metric(
        label=f"Lowest Marriage Rate in Peak CPI Year ({peak_inflation_year})",
        value=f"{lowest_rate_val:.2f}%",
        delta=f"{lowest_rate_row['state']}, {str(lowest_rate_row['sex']).title()} (Peak CPI: {peak_inflation_val:.2f})",
        delta_color="off"
    )

st.divider()

# ---------------------------------------------------------
# Chart 1 & Chart 2: Regression Scatter & Correlation Heatmap
# ---------------------------------------------------------
sns.set_theme(style="whitegrid")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("CPI vs. Marriage Rate Trend")
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.regplot(
        data=df_filtered,
        x="index",
        y="rate",
        scatter_kws={"alpha": 0.6, "color": "#1a73e8"},
        line_kws={"color": "#d93025", "linewidth": 2},
        ax=ax1
    )
    ax1.set_title(f"Relationship Between CPI and Marriage Rate ({selected_years[0]}–{selected_years[1]})", fontsize=12, pad=12)
    ax1.set_xlabel("Consumer Price Index (CPI)", fontsize=10)
    ax1.set_ylabel("Marriage Rate (%)", fontsize=10)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

with chart_col2:
    st.subheader("Socio-Economic Correlation Matrix")
    if len(df_full_filtered) >= 2:
        columns_of_interest = {
            "rate": "Marriage Rate (%)",
            "index": "CPI Index",
            "cpi_growth_yoy": "CPI YoY Growth (%)",
            "income_mean": "Mean Income",
            "income_median": "Median Income"
        }
        df_corr_subset = df_full_filtered[list(columns_of_interest.keys())].rename(columns=columns_of_interest)
        corr_matrix = df_corr_subset.corr()

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            fmt=".3f",
            linewidths=0.5,
            cbar_kws={"label": "Pearson Correlation Coefficient"},
            ax=ax2
        )
        income_years = sorted(df_full_filtered["year"].unique())
        ax2.set_title(f"Correlation Matrix (Income Survey Years: {', '.join(map(str, income_years))})", fontsize=12, pad=12)
        plt.xticks(rotation=35, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)
    else:
        st.info("Not enough merged income survey records for the current filter selection to generate a correlation heatmap.")

st.divider()

# ---------------------------------------------------------
# Chart 3: Multi-Dimensional Bubble Scatter Plot
# ---------------------------------------------------------
st.subheader("Multi-Dimensional View: CPI, Marriage Rate & Household Income")

df_3d = df_full_filtered.dropna(subset=["rate", "index", "income_mean", "income_median"]).reset_index(drop=True)

if not df_3d.empty:
    sns.set_theme(style="white")
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    min_size, max_size = 60, 500
    mean_inc_min = df_3d["income_mean"].min()
    mean_inc_max = df_3d["income_mean"].max()

    if mean_inc_max > mean_inc_min:
        sizes = min_size + (df_3d["income_mean"] - mean_inc_min) / (mean_inc_max - mean_inc_min) * (max_size - min_size)
    else:
        sizes = np.full(len(df_3d), (min_size + max_size) / 2)

    scatter = ax3.scatter(
        x=df_3d["index"],
        y=df_3d["rate"],
        s=sizes,
        c=df_3d["income_median"],
        cmap="viridis",
        alpha=0.75,
        edgecolors="white",
        linewidth=0.8
    )

    # Annotate sampled points with state and year
    step = max(1, len(df_3d) // 15)
    for i, row in df_3d.iterrows():
        if i % step == 0:
            ax3.annotate(
                f"{row['state']} ({int(row['year'])})",
                (row["index"], row["rate"]),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color="#333333"
            )

    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("Median Household Income (RM)", fontsize=10)

    ax3.set_title("Bubble Scatter Plot: CPI, Marriage Rate & Household Income", fontsize=13, pad=14)
    ax3.set_xlabel("Consumer Price Index (CPI Index)", fontsize=10)
    ax3.set_ylabel("Marriage Rate (%)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # Add size legend for Mean Income
    if mean_inc_max > mean_inc_min:
        for size_val in [4000, 8000, 12000]:
            if mean_inc_min <= size_val <= mean_inc_max or len(df_3d) > 2:
                norm_size = np.clip(
                    min_size + (size_val - mean_inc_min) / (mean_inc_max - mean_inc_min) * (max_size - min_size),
                    min_size,
                    max_size
                )
                ax3.scatter([], [], c="gray", alpha=0.5, s=norm_size, label=f"RM {size_val:,}")
        ax3.legend(title="Mean Income (Bubble Size)", loc="upper right", frameon=True)

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)
else:
    st.info("No combined income and marriage records available for the selected filters.")

# ---------------------------------------------------------
# Raw Data Explorer & CSV Export
# ---------------------------------------------------------
with st.expander("View & Download Consolidated Dataset"):
    tab1, tab2 = st.tabs(["CPI & Marriage Data (2018–2024)", "Merged Data with Household Income"])

    with tab1:
        st.dataframe(df_filtered, use_container_width=True)
        st.download_button(
            label="Download CPI & Marriage CSV",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="malaysia_cpi_marriages_2018_2024.csv",
            mime="text/csv"
        )

    with tab2:
        st.dataframe(df_full_filtered, use_container_width=True)
        st.download_button(
            label="Download Full Merged CSV",
            data=df_full_filtered.to_csv(index=False).encode("utf-8"),
            file_name="malaysia_cpi_marriages_income.csv",
            mime="text/csv"
        )
