"""
main.py
=======
CLI entry point for the Climate Trend Analyzer pipeline.
Executes data generation, cleaning, exploratory analysis, trend estimation,
anomaly detection, educational forecasting, and narrative reporting.
"""

import sys
from pathlib import Path
import pandas as pd

# Ensure UTF-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add workspace root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.anomaly_detection import save_anomaly_static_figures, synthesize_anomalies
from src.data_generator import save_synthetic_sample
from src.eda import (
    compute_descriptive_statistics,
    compute_monthly_climatology,
    compute_yearly_summary,
    save_eda_static_figures,
)
from src.forecasting import fit_holt_winters_forecast, train_test_forecast_split
from src.insights import generate_climate_insights
from src.preprocessing import clean_climate_dataset, save_processed_dataset
from src.trend_analysis import compute_rolling_averages, estimate_linear_trend, save_trend_static_figures


def run_pipeline():
    print("=" * 65)
    print("      CLIMATE TREND ANALYZER - AUTOMATED PIPELINE")
    print("=" * 65)

    # 1. Ensure sample dataset exists
    sample_csv = BASE_DIR / "data" / "sample" / "climate_monthly_synthetic.csv"
    if not sample_csv.exists():
        print("[1/6] Generating synthetic climate dataset...")
        save_synthetic_sample(sample_csv)
    else:
        print(f"[1/6] Found dataset: {sample_csv.name}")

    # 2. Data Cleaning & Preprocessing
    print("[2/6] Validating, cleaning, and imputing climate records...")
    df_clean, quality_report = clean_climate_dataset(sample_csv)
    processed_csv = BASE_DIR / "data" / "processed" / "climate_monthly_cleaned.csv"
    save_processed_dataset(df_clean, processed_csv)
    print(f"      Records cleaned: {len(df_clean)} (Duplicates removed: {quality_report['duplicate_timestamps']})")

    # 3. Exploratory Analysis
    print("[3/6] Computing descriptive statistics and climatological baselines...")
    desc_stats = compute_descriptive_statistics(df_clean)
    yearly_df = compute_yearly_summary(df_clean)
    climatology_df = compute_monthly_climatology(df_clean)

    # Export EDA figures and tables
    figures_dir = BASE_DIR / "outputs" / "figures"
    tables_dir = BASE_DIR / "outputs" / "tables"
    reports_dir = BASE_DIR / "outputs" / "reports"
    for d in [figures_dir, tables_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    desc_stats.to_csv(tables_dir / "descriptive_statistics.csv")
    yearly_df.to_csv(tables_dir / "yearly_summary.csv", index=False)
    save_eda_static_figures(df_clean, figures_dir)

    # 4. Trend Analysis
    print("[4/6] Quantifying long-term temperature & precipitation trends...")
    df_rolled = compute_rolling_averages(df_clean, windows=(12, 36))
    trend_stats = estimate_linear_trend(df_rolled, target_col="temperature_c")
    save_trend_static_figures(df_rolled, trend_stats, figures_dir)
    print(f"      Decadal warming rate: {trend_stats['slope_per_decade']:+.2f}°C/decade (R²={trend_stats['r_squared']:.3f}, p={trend_stats['p_value']:.4e})")

    # 5. Anomaly Detection
    print("[5/6] Executing dual-method anomaly detection (Z-score + Isolation Forest)...")
    df_anomalies, bench_metrics = synthesize_anomalies(df_rolled, z_threshold=2.5, contamination=0.03)
    save_anomaly_static_figures(df_anomalies, figures_dir)

    anomaly_table = df_anomalies[df_anomalies["is_anomaly"]][
        ["date", "temperature_c", "rainfall_mm", "humidity_pct", "anomaly_reason"]
    ]
    anomaly_table.to_csv(tables_dir / "anomalies_detected.csv", index=False)
    print(f"      Detected {len(anomaly_table)} anomalous months.")

    # 6. Forecasting & Report Synthesis
    print("[6/6] Fitting educational time-series forecasting & generating report...")
    train_df, test_df = train_test_forecast_split(df_anomalies, test_periods=12)
    _, _, forecast_metrics = fit_holt_winters_forecast(train_df, test_df)

    insights = generate_climate_insights(
        df_clean=df_anomalies,
        yearly_df=yearly_df,
        trend_stats=trend_stats,
        anomaly_df=df_anomalies,
        quality_report=quality_report,
        forecast_metrics=forecast_metrics,
    )

    report_path = reports_dir / "climate_summary_report.md"
    report_path.write_text(insights["markdown_report"], encoding="utf-8")

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"- Cleaned data: {processed_csv}")
    print(f"- Tables: {tables_dir}")
    print(f"- Figures: {figures_dir}")
    print(f"- Summary Report: {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
