"""
insights.py
===========
Automated plain-language narrative synthesis from climate analytics.
Translates statistical metrics, trends, and anomalies into clear,
rigorous findings suitable for portfolios, executives, and research reports.
"""

from typing import Any, Dict, List
import pandas as pd


def generate_climate_insights(
    df_clean: pd.DataFrame,
    yearly_df: pd.DataFrame,
    trend_stats: Dict[str, Any],
    anomaly_df: pd.DataFrame,
    quality_report: Dict[str, Any],
    forecast_metrics: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generate structured, automated narrative insights from climate analytics.

    Parameters
    ----------
    df_clean : pd.DataFrame
        Cleaned records.
    yearly_df : pd.DataFrame
        Annual summary DataFrame.
    trend_stats : dict
        Output from estimate_linear_trend.
    anomaly_df : pd.DataFrame
        Output from synthesize_anomalies.
    quality_report : dict
        Output from audit_data_quality.
    forecast_metrics : dict, optional
        Output from forecasting module.

    Returns
    -------
    dict
        Structured insights with bullet points, key metrics, and markdown text.
    """
    is_synthetic = bool(df_clean.get("is_synthetic", pd.Series([False])).iloc[0])

    # 1. Temperature Trend Insight
    slope_dec = trend_stats.get("slope_per_decade", 0.0)
    p_val = trend_stats.get("p_value", 1.0)
    r2 = trend_stats.get("r_squared", 0.0)
    is_sig = trend_stats.get("is_statistically_significant", False)

    if slope_dec > 0:
        trend_direction = "warming"
        trend_adverb = "statistically significant upward" if is_sig else "slight positive"
    else:
        trend_direction = "cooling"
        trend_adverb = "statistically significant downward" if is_sig else "slight negative"

    trend_narrative = (
        f"The data exhibits a {trend_adverb} {trend_direction} trend at a rate of "
        f"{slope_dec:+.2f}°C per decade (R² = {r2:.3f}, p = {p_val:.4e}). "
        f"{'This trend is statistically significant at the α = 0.05 level.' if is_sig else 'This trend does not meet the standard threshold for statistical significance.'}"
    )

    # 2. Historical Extremes
    hottest_yr_row = yearly_df.loc[yearly_df["mean_temp_c"].idxmax()]
    coolest_yr_row = yearly_df.loc[yearly_df["mean_temp_c"].idxmin()]
    wettest_yr_row = yearly_df.loc[yearly_df["total_rainfall_mm"].idxmax()] if "total_rainfall_mm" in yearly_df.columns else None

    extremes_narrative = (
        f"The warmest recorded calendar year was {int(hottest_yr_row['year'])} with an annual mean temperature "
        f"of {hottest_yr_row['mean_temp_c']:.2f}°C. Conversely, the coolest year was {int(coolest_yr_row['year'])} "
        f"at {coolest_yr_row['mean_temp_c']:.2f}°C."
    )
    if wettest_yr_row is not None:
        extremes_narrative += (
            f" The highest cumulative rainfall occurred in {int(wettest_yr_row['year'])} "
            f"with {wettest_yr_row['total_rainfall_mm']:.1f} mm."
        )

    # 3. Anomaly Detection Insight
    anom_count = int(anomaly_df["is_anomaly"].sum()) if "is_anomaly" in anomaly_df.columns else 0
    anom_pct = round((anom_count / len(anomaly_df) * 100), 2) if len(anomaly_df) > 0 else 0.0

    heatwaves = anomaly_df["anomaly_reason"].str.contains("Heatwave", na=False).sum()
    cold_snaps = anomaly_df["anomaly_reason"].str.contains("Cold Snap", na=False).sum()
    extreme_rains = anomaly_df["anomaly_reason"].str.contains("Extreme Rain", na=False).sum()

    anomaly_narrative = (
        f"A total of {anom_count} anomalous months ({anom_pct}% of total records) were flagged by the "
        f"dual-method consensus system. Identified events include {heatwaves} severe heatwave occurrences, "
        f"{cold_snaps} extreme cold snaps, and {extreme_rains} unseasonal heavy precipitation events."
    )

    # 4. Seasonal Insights
    seasonal_agg = df_clean.groupby("season")["temperature_c"].mean().round(2).to_dict()
    seasonal_narrative = (
        f"Mean temperatures across seasons: Summer averaged {seasonal_agg.get('Summer', 'N/A')}°C, "
        f"Spring {seasonal_agg.get('Spring', 'N/A')}°C, Autumn {seasonal_agg.get('Autumn', 'N/A')}°C, "
        f"and Winter {seasonal_agg.get('Winter', 'N/A')}°C."
    )

    # 5. Scientific Disclaimer & Attribution
    if is_synthetic:
        disclaimer = (
            "**Simulated Data Notice**: This dataset was generated using physical simulation models "
            "with controlled parameters and noise for academic, portfolio, and benchmarking purposes. "
            "It does NOT represent measured climate observations for a real meteorological station."
        )
    else:
        disclaimer = (
            "**Observational Data Notice**: These findings represent empirical observations from the provided dataset. "
            "Correlation between variables does not prove direct causation, and localized historical trends "
            "must be considered alongside broader regional climatological records."
        )

    # Markdown full report
    markdown_report = f"""# Climate Trend Analysis Report

## Executive Summary
- **Data Provenance**: {"Synthetic Simulation (Reproducible Benchmark)" if is_synthetic else "Empirical Climate Records"}
- **Temporal Span**: {quality_report.get('date_start', 'N/A')} to {quality_report.get('date_end', 'N/A')} ({quality_report.get('total_records', 0)} monthly records)
- **Primary Trend**: {trend_direction.capitalize()} at **{slope_dec:+.2f}°C / decade** ({'Significant' if is_sig else 'Not Statistically Significant'})
- **Warmest Year**: {int(hottest_yr_row['year'])} ({hottest_yr_row['mean_temp_c']:.2f}°C)
- **Coolest Year**: {int(coolest_yr_row['year'])} ({coolest_yr_row['mean_temp_c']:.2f}°C)
- **Detected Anomalies**: {anom_count} months ({anom_pct}%)

---

## Key Findings

### 1. Temperature Trajectory
{trend_narrative}

### 2. Climatological Extremes
{extremes_narrative}

### 3. Unusual Observations & Extreme Events
{anomaly_narrative}

### 4. Seasonal Overview
{seasonal_narrative}

---

## Scientific Rigor & Limitations
> {disclaimer}
> 
> *Educational & Methodological Note*: Time-series linear regression and exponential smoothing extrapolate past statistical patterns. They do not simulate atmospheric-ocean general circulation dynamics, volcanic aerosols, or greenhouse gas emission trajectory scenarios (IPCC SSPs).
"""

    return {
        "trend_narrative": trend_narrative,
        "extremes_narrative": extremes_narrative,
        "anomaly_narrative": anomaly_narrative,
        "seasonal_narrative": seasonal_narrative,
        "disclaimer": disclaimer,
        "markdown_report": markdown_report,
        "is_synthetic": is_synthetic,
        "key_metrics": {
            "slope_per_decade": slope_dec,
            "warmest_year": int(hottest_yr_row["year"]),
            "warmest_year_temp": float(hottest_yr_row["mean_temp_c"]),
            "coolest_year": int(coolest_yr_row["year"]),
            "coolest_year_temp": float(coolest_yr_row["mean_temp_c"]),
            "anomalies_detected": anom_count,
        },
    }
