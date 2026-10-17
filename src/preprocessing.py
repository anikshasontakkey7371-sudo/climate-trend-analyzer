"""
preprocessing.py
================
Robust data validation, cleaning, quality auditing, and imputation
for climate observations and synthetic benchmarks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


COLUMN_ALIASES: Dict[str, List[str]] = {
    "date": ["date", "datetime", "timestamp", "time", "date_time", "dt"],
    "temperature_c": [
        "temperature_c",
        "temperature",
        "temp",
        "temp_c",
        "mean_temp",
        "tavg",
        "air_temp",
    ],
    "rainfall_mm": [
        "rainfall_mm",
        "rainfall",
        "rain",
        "precipitation",
        "prcp",
        "precip_mm",
    ],
    "humidity_pct": [
        "humidity_pct",
        "humidity",
        "rh",
        "relative_humidity",
        "rel_humidity",
    ],
}


def get_season(month: int) -> str:
    """Map calendar month to meteorological season."""
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    else:
        return "Autumn"


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize incoming DataFrame column headers to canonical project names:
    'date', 'temperature_c', 'rainfall_mm', 'humidity_pct'.
    """
    df_clean = df.copy()
    col_map = {}

    lower_cols = {str(c).strip().lower(): c for c in df_clean.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        found = False
        for alias in aliases:
            if alias in lower_cols:
                col_map[lower_cols[alias]] = canonical
                found = True
                break
        if not found and canonical in df_clean.columns:
            col_map[canonical] = canonical

    df_clean = df_clean.rename(columns=col_map)
    return df_clean


def audit_data_quality(df: pd.DataFrame) -> Dict[str, Union[int, float, str, Dict]]:
    """
    Generate an informative data quality audit report.

    Parameters
    ----------
    df : pd.DataFrame
        Input climate DataFrame.

    Returns
    -------
    dict
        Data quality metrics including missing counts, percentages, duplicate count,
        and temporal span.
    """
    total_rows = len(df)
    missing_summary = {}

    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        pct_miss = round((n_miss / total_rows * 100), 2) if total_rows > 0 else 0.0
        missing_summary[col] = {"missing_count": n_miss, "missing_pct": pct_miss}

    n_duplicates = int(df.duplicated(subset=["date"]).sum()) if "date" in df.columns else 0

    date_start = str(df["date"].min()) if ("date" in df.columns and not df["date"].isna().all()) else "N/A"
    date_end = str(df["date"].max()) if ("date" in df.columns and not df["date"].isna().all()) else "N/A"

    return {
        "total_records": total_rows,
        "duplicate_timestamps": n_duplicates,
        "date_start": date_start,
        "date_end": date_end,
        "columns_audited": list(df.columns),
        "missing_summary": missing_summary,
    }


def clean_climate_dataset(
    data: Union[pd.DataFrame, str, Path],
    imputation_strategy: str = "linear_interpolation",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate, clean, deduplicate, and impute climate records.

    Parameters
    ----------
    data : pd.DataFrame or str or Path
        Input DataFrame or path to CSV file.
    imputation_strategy : str
        Strategy for missing values:
        - 'linear_interpolation': Linear interpolation over time (default).
        - 'seasonal_median': Impute with monthly calendar median.
        - 'forward_fill': Forward fill followed by backward fill.
        - 'drop': Remove rows with any missing values.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Cleaned, feature-engineered DataFrame and the data quality audit dictionary.
    """
    if isinstance(data, (str, Path)):
        df = pd.read_csv(data)
    else:
        df = data.copy()

    if df.empty:
        raise ValueError("Cannot process empty dataset.")

    # 1. Standardize Column Headers
    df = standardize_column_names(df)

    if "date" not in df.columns:
        raise KeyError(
            f"Required column 'date' not found. Available columns: {list(df.columns)}"
        )
    if "temperature_c" not in df.columns:
        raise KeyError(
            f"Required column 'temperature_c' not found. Available columns: {list(df.columns)}"
        )

    # 2. Date Parsing and Sorting
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values(by="date").reset_index(drop=True)

    # 3. Quality Audit before cleaning
    quality_report = audit_data_quality(df)

    # 4. Handle Duplicate Timestamps (keep first or aggregate mean)
    if quality_report["duplicate_timestamps"] > 0:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
        df = df.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

    # 5. Physical Bound Sanity Checks
    if "rainfall_mm" in df.columns:
        # Precipitation cannot be physically negative
        df["rainfall_mm"] = df["rainfall_mm"].apply(lambda v: max(0.0, v) if pd.notna(v) else v)

    if "humidity_pct" in df.columns:
        # Relative humidity bounded between 0% and 100%
        df["humidity_pct"] = df["humidity_pct"].apply(
            lambda v: min(100.0, max(0.0, v)) if pd.notna(v) else v
        )

    # 6. Imputation of Missing Values
    target_numeric_cols = [
        c for c in ["temperature_c", "rainfall_mm", "humidity_pct"] if c in df.columns
    ]

    if imputation_strategy == "drop":
        df = df.dropna(subset=target_numeric_cols).reset_index(drop=True)
    elif imputation_strategy == "forward_fill":
        df[target_numeric_cols] = df[target_numeric_cols].ffill().bfill()
    elif imputation_strategy == "seasonal_median":
        df["_temp_month"] = df["date"].dt.month
        for col in target_numeric_cols:
            monthly_medians = df.groupby("_temp_month")[col].transform("median")
            df[col] = df[col].fillna(monthly_medians)
            # fallback if all values in a month are NaN
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        df = df.drop(columns=["_temp_month"])
    else:  # default 'linear_interpolation'
        df[target_numeric_cols] = df[target_numeric_cols].interpolate(
            method="linear", limit_direction="both"
        )
        # fallback bfill/ffill for any edge NaNs
        df[target_numeric_cols] = df[target_numeric_cols].bfill().ffill()

    # 7. Add Standardized Calendar Features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["month_name"] = df["date"].dt.strftime("%B")
    df["season"] = df["month"].apply(get_season)

    # Retain ground truth anomaly if present
    if "is_anomaly_ground_truth" not in df.columns:
        df["is_anomaly_ground_truth"] = False
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False

    return df, quality_report


def save_processed_dataset(df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save cleaned climate records to data/processed/."""
    if output_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_path = base_dir / "data" / "processed" / "climate_monthly_cleaned.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
