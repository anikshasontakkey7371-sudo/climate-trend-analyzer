"""
test_analysis.py
================
Automated unit tests for data generator, EDA, trends, anomaly detection,
forecasting, and narrative insights.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    detect_isolation_forest_anomalies,
    detect_statistical_anomalies,
    synthesize_anomalies,
)
from src.data_generator import generate_climate_data
from src.eda import (
    compute_descriptive_statistics,
    compute_monthly_climatology,
    compute_yearly_summary,
)
from src.forecasting import fit_holt_winters_forecast, train_test_forecast_split
from src.insights import generate_climate_insights
from src.trend_analysis import compute_rolling_averages, estimate_linear_trend


@pytest.fixture
def clean_synthetic_data():
    """Fixture providing clean 10-year monthly synthetic data."""
    df = generate_climate_data(
        start_year=2000,
        end_year=2015,
        freq="MS",
        missing_fraction=0.0,
        random_seed=123,
    )
    return df


def test_data_generator_reproducibility():
    """Test synthetic data generator produces identical results with fixed seed."""
    df1 = generate_climate_data(start_year=2000, end_year=2005, random_seed=42)
    df2 = generate_climate_data(start_year=2000, end_year=2005, random_seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_data_generator_columns(clean_synthetic_data):
    """Test presence of required physics-informed columns and synthetic flag."""
    expected_cols = [
        "date", "temperature_c", "rainfall_mm", "humidity_pct",
        "year", "month", "season", "is_synthetic", "is_anomaly_ground_truth"
    ]
    for col in expected_cols:
        assert col in clean_synthetic_data.columns
    assert clean_synthetic_data["is_synthetic"].all() is True or clean_synthetic_data["is_synthetic"].all() == True


def test_eda_summaries(clean_synthetic_data):
    """Test yearly and climatology summary aggregations."""
    yearly = compute_yearly_summary(clean_synthetic_data)
    assert len(yearly) == 16  # 2000 through 2015
    assert "mean_temp_c" in yearly.columns

    climatology = compute_monthly_climatology(clean_synthetic_data)
    assert len(climatology) == 12
    assert (climatology["temp_mean"] > 0).all()


def test_trend_analysis_linear_regression(clean_synthetic_data):
    """Test linear trend estimation returns valid stats."""
    trend_stats = estimate_linear_trend(clean_synthetic_data, target_col="temperature_c")
    assert "slope_per_decade" in trend_stats
    assert "r_squared" in trend_stats
    assert "p_value" in trend_stats
    assert 0 <= trend_stats["r_squared"] <= 1
    assert 0 <= trend_stats["p_value"] <= 1


def test_anomaly_detection_pipeline(clean_synthetic_data):
    """Test statistical and Isolation Forest anomaly detection execution."""
    df_anom, bench = synthesize_anomalies(clean_synthetic_data, z_threshold=2.5, contamination=0.03)
    assert "is_anomaly" in df_anom.columns
    assert "anomaly_reason" in df_anom.columns
    # Confirm some anomalies are detected
    assert df_anom["is_anomaly"].sum() > 0


def test_forecasting_holt_winters(clean_synthetic_data):
    """Test Holt-Winters training, test validation, and future projection."""
    train_df, test_df = train_test_forecast_split(clean_synthetic_data, test_periods=12)
    test_preds, future_preds, metrics = fit_holt_winters_forecast(
        train_df, test_df, forecast_horizon=12, seasonal_periods=12
    )
    assert len(test_preds) == 12
    assert len(future_preds) == 12
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0


def test_narrative_insights_generation(clean_synthetic_data):
    """Test automated markdown and narrative synthesis."""
    yearly = compute_yearly_summary(clean_synthetic_data)
    trend_stats = estimate_linear_trend(clean_synthetic_data)
    df_anom, _ = synthesize_anomalies(clean_synthetic_data)
    quality = {"date_start": "2000-01-01", "date_end": "2015-12-01", "total_records": len(clean_synthetic_data)}

    insights = generate_climate_insights(
        df_clean=clean_synthetic_data,
        yearly_df=yearly,
        trend_stats=trend_stats,
        anomaly_df=df_anom,
        quality_report=quality,
    )

    assert "trend_narrative" in insights
    assert "markdown_report" in insights
    assert "Executive Summary" in insights["markdown_report"]
    assert "Simulated Data Notice" in insights["markdown_report"]
