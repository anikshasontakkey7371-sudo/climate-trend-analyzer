"""
dashboard.py
============
Interactive Streamlit Dashboard for Climate Trend Analyzer.
Provides exploratory analytics, trend quantification, dual-method anomaly detection,
educational forecasting, and automated scientific insights.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Climate Trend Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.anomaly_detection import create_anomaly_scatter_chart, synthesize_anomalies
from src.data_generator import generate_climate_data, save_synthetic_sample
from src.eda import (
    compute_correlation_matrix,
    compute_descriptive_statistics,
    compute_monthly_climatology,
    compute_seasonal_summary,
    compute_yearly_summary,
    create_correlation_heatmap,
    create_monthly_climatology_chart,
    create_seasonal_boxplot,
)
from src.forecasting import (
    create_forecast_chart,
    fit_holt_winters_forecast,
    generate_forecast_dates,
    train_test_forecast_split,
)
from src.insights import generate_climate_insights
from src.preprocessing import clean_climate_dataset
from src.trend_analysis import (
    compute_monthly_temperature_anomalies,
    compute_rolling_averages,
    create_annual_anomaly_bars,
    create_rainfall_trend_chart,
    create_temperature_trend_chart,
    estimate_linear_trend,
)

# Custom CSS for polished scientific styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 13px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-sim {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    .badge-obs {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # ------------------ SIDEBAR ------------------
    st.sidebar.title("🌍 Climate Analyzer")
    st.sidebar.caption("Student Portfolio & Technical Analytics Platform")

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Data Source")
    data_source_mode = st.sidebar.radio(
        "Select Data Input:",
        ["Sample Synthetic Dataset", "Custom Synthetic Parameters", "Upload Climate CSV"],
    )

    df_raw = None
    if data_source_mode == "Sample Synthetic Dataset":
        sample_path = PROJECT_ROOT / "data" / "sample" / "climate_monthly_synthetic.csv"
        if not sample_path.exists():
            sample_path = save_synthetic_sample(sample_path)
        df_raw = pd.read_csv(sample_path)
        st.sidebar.success("Loaded pre-computed 1990-2023 monthly sample.")

    elif data_source_mode == "Custom Synthetic Parameters":
        start_year = st.sidebar.slider("Start Year", 1970, 2010, 1990)
        end_year = st.sidebar.slider("End Year", 2015, 2030, 2023)
        trend_rate = st.sidebar.slider("Warming Trend (°C/decade)", -0.5, 1.0, 0.28, 0.02)
        noise_level = st.sidebar.slider("Noise Std Dev (°C)", 0.2, 2.5, 1.1, 0.1)
        anom_rate = st.sidebar.slider("Anomaly Frequency (%)", 0.0, 8.0, 3.0, 0.5) / 100.0
        seed = st.sidebar.number_input("Random Seed", value=42, step=1)

        df_raw = generate_climate_data(
            start_year=start_year,
            end_year=end_year,
            temp_trend_per_decade=trend_rate,
            noise_std_temp=noise_level,
            anomaly_fraction=anom_rate,
            random_seed=seed,
        )
        st.sidebar.info(f"Generated {len(df_raw)} records ({start_year}-{end_year}).")

    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain date and temp)", type=["csv"])
        if uploaded_file is not None:
            df_raw = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.sidebar.warning("Please upload a CSV file or switch to synthetic data.")
            st.info("👋 Welcome! Select a dataset from the sidebar to begin analysis.")
            st.stop()

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Preprocessing Options")
    imputation_strategy = st.sidebar.selectbox(
        "Missing-Value Imputation:",
        ["linear_interpolation", "seasonal_median", "forward_fill", "drop"],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Anomaly Parameters")
    z_cutoff = st.sidebar.slider("Z-Score Cutoff (σ)", 1.5, 4.0, 2.5, 0.1)
    iso_contamination = st.sidebar.slider("Isolation Forest Contamination", 0.01, 0.10, 0.03, 0.01)
    consensus_rule = st.sidebar.radio("Consensus Rule:", ["Union (Any)", "Intersection (Both)"])
    rule_key = "both" if "Both" in consensus_rule else "any"

    # ------------------ PROCESSING PIPELINE ------------------
    try:
        df_clean, quality_report = clean_climate_dataset(df_raw, imputation_strategy=imputation_strategy)
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        st.stop()

    df_clean = compute_monthly_temperature_anomalies(df_clean)
    df_rolled = compute_rolling_averages(df_clean, windows=(12, 36))
    trend_stats = estimate_linear_trend(df_rolled, target_col="temperature_c")
    df_anomalies, bench_metrics = synthesize_anomalies(
        df_rolled,
        z_threshold=z_cutoff,
        contamination=iso_contamination,
        consensus_rule=rule_key,
    )
    yearly_df = compute_yearly_summary(df_clean)
    climatology_df = compute_monthly_climatology(df_clean)
    corr_matrix = compute_correlation_matrix(df_clean)

    # ------------------ MAIN HEADER ------------------
    is_syn = bool(df_clean.get("is_synthetic", pd.Series([False])).iloc[0])
    badge_html = (
        '<span class="badge-sim">SIMULATED BENCHMARK DATA</span>'
        if is_syn
        else '<span class="badge-obs">OBSERVED CLIMATE DATA</span>'
    )

    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h1 style="margin: 0;">Climate Trend Analyzer</h1>
            <div>{badge_html}</div>
        </div>
        <p style="color: #64748b; font-size: 16px; margin-bottom: 20px;">
            A rigorous, physics-informed data science platform for historical temperature trajectory,
            precipitation patterns, dual-method anomaly detection, and educational forecasting.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ------------------ TOP KPI METRICS ------------------
    hottest_yr = yearly_df.loc[yearly_df["mean_temp_c"].idxmax()]
    coolest_yr = yearly_df.loc[yearly_df["mean_temp_c"].idxmin()]
    anom_total = int(df_anomalies["is_anomaly"].sum())

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{quality_report['total_records']}</div>
                <div class="metric-label">Monthly Records</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{trend_stats['slope_per_decade']:+.2f}°C</div>
                <div class="metric-label">Rate / Decade</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{int(hottest_yr['year'])}</div>
                <div class="metric-label">Warmest Year ({hottest_yr['mean_temp_c']:.1f}°C)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{int(coolest_yr['year'])}</div>
                <div class="metric-label">Coolest Year ({coolest_yr['mean_temp_c']:.1f}°C)</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi5:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-value">{anom_total}</div>
                <div class="metric-label">Anomalies Detected</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------ APPLICATION TABS ------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Overview & Data Quality",
        "📈 Temperature & Rainfall Trends",
        "🌦️ Seasonal Patterns",
        "⚠️ Anomaly Detection",
        "🔮 Educational Forecast",
        "💡 Summary Insights & Exports",
    ])

    # --- TAB 1: OVERVIEW & DATA QUALITY ---
    with tab1:
        st.subheader("Data Quality Audit & Raw Record Preview")
        col_q1, col_q2 = st.columns([1, 2])

        with col_q1:
            st.markdown("#### Quality Audit Summary")
            st.write(f"**Date Range:** {quality_report['date_start'][:10]} to {quality_report['date_end'][:10]}")
            st.write(f"**Duplicate Timestamps Removed:** {quality_report['duplicate_timestamps']}")
            st.write(f"**Imputation Strategy Applied:** `{imputation_strategy}`")

            # Missing values summary
            miss_df = pd.DataFrame(quality_report["missing_summary"]).T
            st.dataframe(miss_df, use_container_width=True)

        with col_q2:
            st.markdown("#### Cleaned Dataset Preview (First 8 Rows)")
            st.dataframe(df_clean.head(8), use_container_width=True)

        st.markdown("#### Parametric Descriptive Statistics")
        desc_stats = compute_descriptive_statistics(df_clean)
        st.dataframe(desc_stats, use_container_width=True)

    # --- TAB 2: TRENDS ---
    with tab2:
        st.subheader("Long-Term Climatological Trajectory")
        temp_trend_fig = create_temperature_trend_chart(df_anomalies, trend_stats)
        st.plotly_chart(temp_trend_fig, use_container_width=True)

        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("#### Annual Temperature Anomalies")
            bar_fig = create_annual_anomaly_bars(yearly_df)
            st.plotly_chart(bar_fig, use_container_width=True)

        with col_t2:
            st.markdown("#### Monthly Rainfall & Rolling Baseline")
            rain_fig = create_rainfall_trend_chart(df_clean)
            st.plotly_chart(rain_fig, use_container_width=True)

        # Statistical Metrics Box
        st.markdown("#### Trend Regression Statistics")
        st.info(
            f"**Decadal Slope:** {trend_stats['slope_per_decade']:+.3f}°C/decade | "
            f"**R²:** {trend_stats['r_squared']:.4f} | "
            f"**p-value:** {trend_stats['p_value']:.4e} "
            f"({'Statistically Significant (p < 0.05)' if trend_stats['is_statistically_significant'] else 'Not Statistically Significant'}) | "
            f"**Mann-Kendall Tau:** {trend_stats['mann_kendall_tau']} (p = {trend_stats['mann_kendall_p_value']:.4e})"
        )

    # --- TAB 3: SEASONAL ANALYSIS ---
    with tab3:
        st.subheader("Annual Cycles & Seasonal Distributions")
        climatology_fig = create_monthly_climatology_chart(climatology_df)
        st.plotly_chart(climatology_fig, use_container_width=True)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            box_fig = create_seasonal_boxplot(df_clean)
            st.plotly_chart(box_fig, use_container_width=True)

        with col_s2:
            corr_fig = create_correlation_heatmap(corr_matrix)
            st.plotly_chart(corr_fig, use_container_width=True)

        st.markdown("#### Meteorological Seasonal Aggregates")
        seasonal_df = compute_seasonal_summary(df_clean)
        st.dataframe(seasonal_df, use_container_width=True)

    # --- TAB 4: ANOMALY DETECTION ---
    with tab4:
        st.subheader("Dual-Method Anomaly Detection (Statistical + Machine Learning)")
        st.write(
            "Identifies unseasonal extreme temperatures, prolonged heatwaves, sudden cold snaps, and storm precipitation. "
            "Combines **Monthly Climatological Z-Score** and **Isolation Forest** unsupervised tree ensembles."
        )

        anom_fig = create_anomaly_scatter_chart(df_anomalies)
        st.plotly_chart(anom_fig, use_container_width=True)

        if bench_metrics:
            st.markdown("#### Benchmark Validation Against Synthetic Ground Truth")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision", f"{bench_metrics['precision']:.3f}")
            m2.metric("Recall", f"{bench_metrics['recall']:.3f}")
            m3.metric("F1 Score", f"{bench_metrics['f1_score']:.3f}")
            m4.metric("Injected vs Detected", f"{bench_metrics['ground_truth_count']} / {bench_metrics['detected_count']}")

        st.markdown("#### Identified Anomaly Events Log")
        anom_table = df_anomalies[df_anomalies["is_anomaly"]][
            ["date", "temperature_c", "rainfall_mm", "humidity_pct", "z_score_temp", "iforest_severity", "anomaly_reason"]
        ].sort_values(by="date", ascending=False)
        st.dataframe(anom_table, use_container_width=True)

    # --- TAB 5: FORECASTING ---
    with tab5:
        st.subheader("Educational Time-Series Forecast (Holt-Winters Exponential Smoothing)")
        st.warning(
            "**Scientific Disclaimer**: This statistical extrapolation reflects historical time-series patterns. "
            "It is designed for educational demonstrations and does NOT substitute for physical coupled general circulation models (GCMs) "
            "utilizing greenhouse gas concentration pathways (IPCC SSPs)."
        )

        f_horizon = st.slider("Future Projection Horizon (Months):", 6, 48, 24, 6)

        try:
            train_df, test_df = train_test_forecast_split(df_anomalies, test_periods=12)
            test_preds, future_preds, f_metrics = fit_holt_winters_forecast(
                train_df,
                test_df,
                forecast_horizon=f_horizon,
            )
            future_dates = generate_forecast_dates(df_anomalies["date"].iloc[-1], horizon=f_horizon)

            f_fig = create_forecast_chart(
                train_df=train_df,
                test_df=test_df,
                test_preds=test_preds,
                future_preds=future_preds,
                future_dates=future_dates,
                metrics=f_metrics,
            )
            st.plotly_chart(f_fig, use_container_width=True)

            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            col_f1.metric("Test MAE", f"{f_metrics['MAE']:.2f} °C")
            col_f2.metric("Test RMSE", f"{f_metrics['RMSE']:.2f} °C")
            col_f3.metric("Test MAPE", f"{f_metrics['MAPE_pct']:.2f} %")
            col_f4.metric("Hold-out R²", f"{f_metrics['R2']:.3f}")
        except Exception as err:
            st.error(f"Forecasting unavailable for current dataset slice: {err}")

    # --- TAB 6: INSIGHTS & EXPORTS ---
    with tab6:
        st.subheader("Automated Executive Summary & Findings")
        insights = generate_climate_insights(
            df_clean=df_anomalies,
            yearly_df=yearly_df,
            trend_stats=trend_stats,
            anomaly_df=df_anomalies,
            quality_report=quality_report,
        )

        st.markdown(insights["markdown_report"])

        st.markdown("---")
        st.subheader("📥 Export Center")
        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            csv_clean = df_anomalies.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Cleaned Data (CSV)",
                data=csv_clean,
                file_name="cleaned_climate_data.csv",
                mime="text/csv",
            )

        with col_d2:
            csv_anom = anom_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Anomalies Log (CSV)",
                data=csv_anom,
                file_name="detected_climate_anomalies.csv",
                mime="text/csv",
            )

        with col_d3:
            report_md = insights["markdown_report"].encode("utf-8")
            st.download_button(
                label="Download Executive Report (MD)",
                data=report_md,
                file_name="climate_analysis_report.md",
                mime="text/markdown",
            )


if __name__ == "__main__":
    main()
