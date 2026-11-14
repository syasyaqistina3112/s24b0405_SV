import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit page configuration
st.set_page_config(
    page_title="Malaysia Marriage, Household Income & CPI Dashboard",
    layout="wide"
)

st.title("📊 Malaysia Marriage Rates, Household Income & CPI Analysis (2018–2024)")
st.markdown("Data Source: **Department of Statistics Malaysia (DOSM)**")
st.info("Objective 1: To examine the trends in marriage rates, household income, and consumer prices over the years.")

# 1. Load and process data (cached for faster performance)
@st.cache_data
def load_and_process_data():
    url_mar = 'https://storage.dosm.gov.my/demography/marriages_state.parquet'
    url_inc = 'https://storage.dosm.gov.my/hies/hh_income_state.parquet'
    url_cpi = 'https://storage.dosm.gov.my/cpi/cpi_2d_state.parquet'

    df_mar = pd.read_parquet(url_mar)
    df_income = pd.read_parquet(url_inc)
    df_cpi = pd.read_parquet(url_cpi)

    for df in [df_mar, df_income, df_cpi]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

    # Filter records from 2018 to 2024 inclusive
    start_date = '2018-01-01'
    end_date = '2024-12-31'

    df_mar_filtered = df_mar[(df_mar['date'] >= start_date) & (df_mar['date'] <= end_date)].copy()
    df_income_filtered = df_income[(df_income['date'] >= start_date) & (df_income['date'] <= end_date)].copy()
    df_cpi_filtered = df_cpi[(df_cpi['date'] >= start_date) & (df_cpi['date'] <= end_date)].copy()

    # Standardize year column
    df_mar_filtered['year'] = df_mar_filtered['date'].dt.year
    df_income_filtered['year'] = df_income_filtered['date'].dt.year

    # Process CPI data (filter 'overall' division and compute annual mean)
    df_cpi_overall = df_cpi_filtered[df_cpi_filtered['division'] == 'overall'].copy()
    df_cpi_overall['year'] = df_cpi_overall['date'].dt.year
    df_cpi_annual = df_cpi_overall.groupby(['state', 'year'])['index'].mean().reset_index()

    # Pivot marriage data by sex to ensure state-year uniqueness
    df_mar_pivot = df_mar_filtered.pivot_table(
        index=['state', 'year'],
        columns='sex',
        values=['abs', 'rate']
    ).reset_index()

    df_mar_pivot.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in df_mar_pivot.columns
    ]

    # Merge the three datasets
    df_merged = pd.merge(
        df_mar_pivot,
        df_income_filtered.drop(columns=['date']),
        on=['state', 'year'],
        how='outer'
    )
    df_merged = pd.merge(df_merged, df_cpi_annual, on=['state', 'year'], how='outer')

    df_merged['date'] = pd.to_datetime(df_merged['year'].astype(str) + '-01-01')
    df_merged = df_merged.sort_values(['state', 'year']).reset_index(drop=True)

    return df_merged

# Load data with a loading spinner
with st.spinner("Fetching and processing data from DOSM..."):
    df_merged = load_and_process_data()

# --- SIDEBAR: STATE FILTER ---
st.sidebar.header("Data Filters")
state_list = sorted(df_merged['state'].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select State(s) (Leave empty for all states):",
    options=state_list,
    default=[]
)

if selected_states:
    df_view = df_merged[df_merged['state'].isin(selected_states)]
else:
    df_view = df_merged.copy()

# --- SECTION 1: KEY METRICS (st.metric) ---
st.subheader("Key Summary Metrics")

df_2024 = df_view[df_view['year'] == 2024]

# Marriage Rates
latest_marriage_female = df_2024['rate_female'].mean()
latest_marriage_male = df_2024['rate_male'].mean()
avg_marriage_female = df_view['rate_female'].mean()
avg_marriage_male = df_view['rate_male'].mean()

# Household Income (automatically selects the latest available survey year)
df_inc_valid = df_view.dropna(subset=['income_mean'])
latest_inc_year = int(df_inc_valid['year'].max()) if not df_inc_valid.empty else 2024
df_latest_inc = df_view[df_view['year'] == latest_inc_year]
latest_income_mean = df_latest_inc['income_mean'].mean()
latest_income_median = df_latest_inc['income_median'].mean()

# Consumer Price Index (CPI)
latest_cpi = df_2024['index'].mean()

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="Female Marriage Rate (2024)",
        value=f"{latest_marriage_female:.2f}" if pd.notna(latest_marriage_female) else "N/A",
        delta=f"Avg: {avg_marriage_female:.2f}" if pd.notna(avg_marriage_female) else None,
        delta_color="off"
    )

with col2:
    st.metric(
        label="Male Marriage Rate (2024)",
        value=f"{latest_marriage_male:.2f}" if pd.notna(latest_marriage_male) else "N/A",
        delta=f"Avg: {avg_marriage_male:.2f}" if pd.notna(avg_marriage_male) else None,
        delta_color="off"
    )

with col3:
    st.metric(
        label=f"Mean Household Income ({latest_inc_year})",
        value=f"RM {latest_income_mean:,.2f}" if pd.notna(latest_income_mean) else "N/A"
    )

with col4:
    st.metric(
        label=f"Median Household Income ({latest_inc_year})",
        value=f"RM {latest_income_median:,.2f}" if pd.notna(latest_income_median) else "N/A"
    )

with col5:
    st.metric(
        label="CPI Index (2024)",
        value=f"{latest_cpi:.2f}" if pd.notna(latest_cpi) else "N/A"
    )

# Interactive dataframe viewer (replaces Colab's display())
with st.expander("View Merged Dataset"):
    st.dataframe(df_view, use_container_width=True)

st.divider()

# --- SECTION 2: VISUALIZATIONS ---
sns.set_theme(style="whitegrid")

annual_summary = df_view.groupby('year').agg({
    'rate_female': 'mean',
    'rate_male': 'mean',
    'income_mean': 'mean',
    'income_median': 'mean',
    'index': 'mean'
}).reset_index()

tab1, tab2, tab3 = st.tabs([
    "📈 Annual Trends (Line)",
    "📊 Gender Comparison (Bar)",
    "🗺️ State Breakdown (Heatmap)"
])

# Visualization 1: Line Plots
with tab1:
    st.subheader("Marriage Rates, Household Income & CPI Trends (2018–2024)")
    fig1, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

    sns.lineplot(data=annual_summary, x='year', y='rate_female', ax=axes[0], marker='o', label='Female', color='purple', linewidth=2.5)
    sns.lineplot(data=annual_summary, x='year', y='rate_male', ax=axes[0], marker='s', label='Male', color='teal', linewidth=2.5)
    axes[0].set_title('Average Marriage Rates Trend (2018–2024)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Rate (Per 1,000 population)', fontsize=10)
    axes[0].legend(title='Sex')

    sns.lineplot(data=annual_summary, x='year', y='income_mean', ax=axes[1], marker='o', label='Mean Income', color='forestgreen', linewidth=2.5)
    sns.lineplot(data=annual_summary, x='year', y='income_median', ax=axes[1], marker='^', label='Median Income', color='darkorange', linewidth=2.5)
    axes[1].set_title('Average Monthly Household Income Trend (2018–2024)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Income (RM)', fontsize=10)
    axes[1].legend(title='Metric')

    sns.lineplot(data=annual_summary, x='year', y='index', ax=axes[2], marker='o', color='crimson', linewidth=2.5, label='CPI Index')
    axes[2].set_title('Average Consumer Price Index (CPI) Trend (2018–2024)', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('CPI Index', fontsize=10)
    axes[2].set_xlabel('Year', fontsize=10)

    plt.xticks(annual_summary['year'])
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

# Visualization 2: Grouped Bar Chart
with tab2:
    st.subheader("Average Marriage Rates by Year and Sex")
    df_bar_data = annual_summary.melt(
        id_vars=['year'],
        value_vars=['rate_female', 'rate_male'],
        var_name='Sex',
        value_name='Marriage Rate'
    )
    df_bar_data['Sex'] = df_bar_data['Sex'].map({
        'rate_female': 'Female',
        'rate_male': 'Male'
    })

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_bar_data,
        x='year',
        y='Marriage Rate',
        hue='Sex',
        palette=['orchid', 'cadetblue'],
        ax=ax2
    )
    ax2.set_title('Average Marriage Rates by Year (2018–2024)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Average Marriage Rate (per 1,000 population)', fontsize=12)
    ax2.legend(title='Sex', loc='upper right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# Visualization 3: Heatmap
with tab3:
    st.subheader("Female Marriage Rates Breakdown by State")
    df_heatmap_data = df_view.dropna(subset=['state']).pivot_table(
        index='state',
        columns='year',
        values='rate_female',
        aggfunc='mean'
    )

    if not df_heatmap_data.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            df_heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="Purples",
            linewidths=.5,
            cbar_kws={'label': 'Female Marriage Rate (per 1,000)'},
            ax=ax3
        )
        ax3.set_title('Female Marriage Rates Breakdown by State (2018–2024)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('State', fontsize=12)
        ax3.set_xlabel('Year', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    else:
        st.warning("Not enough data available to display the heatmap.")
