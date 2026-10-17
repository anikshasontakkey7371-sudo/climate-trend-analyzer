"""
test_preprocessing.py
=====================
Automated unit tests for data cleaning, validation, and imputation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    audit_data_quality,
    clean_climate_dataset,
    standardize_column_names,
)


@pytest.fixture
def sample_raw_df():
    """Fixture providing a mock uncleaned climate DataFrame."""
    return pd.DataFrame(
        {
            "Date": ["2020-01-01", "2020-02-01", "2020-03-01", "2020-03-01", "2020-04-01"],
            "Temp": [10.5, np.nan, 15.2, 15.2, 18.0],
            "Rainfall": [50.0, 45.0, -5.0, -5.0, 30.0],  # negative rain to test clipping
            "Humidity": [60.0, 65.0, 110.0, 110.0, 55.0],  # >100% to test bounding
        }
    )


def test_standardize_column_names(sample_raw_df):
    """Test column headers are mapped to canonical names."""
    standardized = standardize_column_names(sample_raw_df)
    assert "date" in standardized.columns
    assert "temperature_c" in standardized.columns
    assert "rainfall_mm" in standardized.columns
    assert "humidity_pct" in standardized.columns


def test_clean_climate_dataset_deduplication(sample_raw_df):
    """Test duplicate dates are dropped and sorted."""
    df_clean, report = clean_climate_dataset(sample_raw_df)
    assert report["duplicate_timestamps"] == 1
    assert len(df_clean) == 4
    # Ensure chronological order
    assert list(df_clean["date"]) == sorted(list(df_clean["date"]))


def test_clean_climate_dataset_physical_bounds(sample_raw_df):
    """Test negative rainfall is clipped to 0 and humidity clamped to <= 100."""
    df_clean, _ = clean_climate_dataset(sample_raw_df)
    assert (df_clean["rainfall_mm"] >= 0).all()
    assert (df_clean["humidity_pct"] <= 100).all()
    assert (df_clean["humidity_pct"] >= 0).all()


def test_clean_climate_dataset_imputation(sample_raw_df):
    """Test missing values are imputed properly without NaNs remaining."""
    df_clean, _ = clean_climate_dataset(sample_raw_df, imputation_strategy="linear_interpolation")
    assert df_clean["temperature_c"].isna().sum() == 0
    # Imputed value between 10.5 and 15.2
    assert 10.5 < df_clean.loc[1, "temperature_c"] < 15.2


def test_clean_climate_dataset_missing_date_column_raises():
    """Test that missing required date column raises KeyError."""
    bad_df = pd.DataFrame({"temp": [12.0, 14.0], "rain": [20.0, 30.0]})
    with pytest.raises(KeyError):
        clean_climate_dataset(bad_df)


def test_clean_climate_dataset_empty_raises():
    """Test that an empty DataFrame raises ValueError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        clean_climate_dataset(empty_df)


def test_audit_data_quality():
    """Test quality audit metrics."""
    test_df = pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=5, freq="D"),
        "temperature_c": [12.0, np.nan, 14.0, 15.0, np.nan],
    })
    report = audit_data_quality(test_df)
    assert report["total_records"] == 5
    assert report["missing_summary"]["temperature_c"]["missing_count"] == 2
    assert report["missing_summary"]["temperature_c"]["missing_pct"] == 40.0
