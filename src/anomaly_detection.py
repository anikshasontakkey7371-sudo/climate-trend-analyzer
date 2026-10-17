"""
anomaly_detection.py
====================
Detects unusual climatic events using dual methodologies:
1. Statistical Climatological Z-score / IQR thresholding.
2. Unsupervised Machine Learning (Isolation Forest).

Provides consensus flagging, severity scoring, explanatory labels,
and evaluation metrics against ground-truth synthetic benchmarks.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score


def detect_statistical_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
    method: str = "monthly_zscore",
) -> pd.DataFrame:
    """
    Detect statistical outliers relative to month-specific climatology.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'month', 'temperature_c', and 'rainfall_mm'.
    z_threshold : float
        Standard deviation cutoff (default 2.5).
    method : str
        'monthly_zscore' or 'iqr'.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'z_score_temp', 'z_score_rain', and 'is_stat_anomaly'.
    """
    res = df.copy()

    # Calculate monthly climatological mean and std
    monthly_stats = res.groupby("month").agg(
        temp_mean=("temperature_c", "mean"),
        temp_std=("temperature_c", "std"),
        rain_mean=("rainfall_mm", "mean") if "rainfall_mm" in res.columns else ("temperature_c", "mean"),
        rain_std=("rainfall_mm", "std") if "rainfall_mm" in res.columns else ("temperature_c", "std"),
    )

    res["climatological_temp_mean"] = res["month"].map(monthly_stats["temp_mean"])
    res["climatological_temp_std"] = res["month"].map(monthly_stats["temp_std"]).replace(0, 1e-6)

    res["z_score_temp"] = (
        (res["temperature_c"] - res["climatological_temp_mean"]) / res["climatological_temp_std"]
    ).round(2)

    if "rainfall_mm" in res.columns:
        res["climatological_rain_mean"] = res["month"].map(monthly_stats["rain_mean"])
        res["climatological_rain_std"] = res["month"].map(monthly_stats["rain_std"]).replace(0, 1e-6)
        res["z_score_rain"] = (
            (res["rainfall_mm"] - res["climatological_rain_mean"]) / res["climatological_rain_std"]
        ).round(2)
    else:
        res["z_score_rain"] = 0.0

    if method == "iqr":
        # Seasonal IQR
        q1 = res.groupby("month")["temperature_c"].transform("quantile", 0.25)
        q3 = res.groupby("month")["temperature_c"].transform("quantile", 0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        res["is_stat_anomaly"] = (res["temperature_c"] < lower_bound) | (res["temperature_c"] > upper_bound)
    else:
        # Z-score cutoff on temperature or severe rainfall
        res["is_stat_anomaly"] = (
            (res["z_score_temp"].abs() >= z_threshold) | (res["z_score_rain"] >= (z_threshold + 0.5))
        )

    return res


def detect_isolation_forest_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect multivariate climate anomalies using Scikit-Learn's Isolation Forest.

    Parameters
    ----------
    df : pd.DataFrame
        Input climate DataFrame.
    contamination : float
        Expected proportion of outliers in the data.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'iforest_score' and 'is_iforest_anomaly'.
    """
    res = df.copy()
    feature_cols = [c for c in ["temperature_c", "rainfall_mm", "humidity_pct"] if c in res.columns]

    # Cyclical month features to capture seasonality
    res["sin_month"] = np.sin(2 * np.pi * res["month"] / 12.0)
    res["cos_month"] = np.cos(2 * np.pi * res["month"] / 12.0)
    features_to_fit = feature_cols + ["sin_month", "cos_month"]

    X = res[features_to_fit].fillna(res[features_to_fit].median())

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=150,
        max_samples="auto",
    )
    preds = iso.fit_predict(X)
    scores = iso.decision_function(X)

    # In Scikit-Learn: -1 indicates outlier, 1 indicates inlier
    res["is_iforest_anomaly"] = preds == -1
    # Lower decision function indicates more abnormal; convert to severity 0 to 1
    min_s, max_s = scores.min(), scores.max()
    res["iforest_severity"] = np.round(1.0 - ((scores - min_s) / (max_s - min_s + 1e-9)), 3)

    res = res.drop(columns=["sin_month", "cos_month"])
    return res


def synthesize_anomalies(
    df: pd.DataFrame,
    z_threshold: float = 2.5,
    contamination: float = 0.03,
    consensus_rule: str = "any",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Run both detection engines, synthesize consensus, and assign human-readable descriptions.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned climate DataFrame.
    z_threshold : float
        Z-score threshold.
    contamination : float
        Isolation forest contamination rate.
    consensus_rule : str
        'any' (union) or 'both' (intersection).

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Annotated DataFrame and evaluation benchmark metrics.
    """
    df_stat = detect_statistical_anomalies(df, z_threshold=z_threshold)
    df_all = detect_isolation_forest_anomalies(df_stat, contamination=contamination)

    if consensus_rule == "both":
        df_all["is_anomaly"] = df_all["is_stat_anomaly"] & df_all["is_iforest_anomaly"]
    else:
        df_all["is_anomaly"] = df_all["is_stat_anomaly"] | df_all["is_iforest_anomaly"]

    # Generate plain-English reason
    reasons = []
    for _, row in df_all.iterrows():
        if not row["is_anomaly"]:
            reasons.append("Normal")
        else:
            tags = []
            if row["z_score_temp"] >= z_threshold:
                tags.append(f"Heatwave (+{row['z_score_temp']:.1f}σ)")
            elif row["z_score_temp"] <= -z_threshold:
                tags.append(f"Cold Snap ({row['z_score_temp']:.1f}σ)")
            if row["z_score_rain"] >= z_threshold:
                tags.append(f"Extreme Rain (+{row['z_score_rain']:.1f}σ)")
            if row["is_iforest_anomaly"] and not tags:
                tags.append(f"Multivariate Pattern Anomaly (Sev: {row['iforest_severity']:.2f})")
            reasons.append("; ".join(tags) if tags else "Statistical Outlier")

    df_all["anomaly_reason"] = reasons

    # Benchmark metrics if synthetic ground truth exists
    benchmark_metrics = {}
    if "is_anomaly_ground_truth" in df_all.columns and df_all["is_anomaly_ground_truth"].any():
        y_true = df_all["is_anomaly_ground_truth"].values
        y_pred = df_all["is_anomaly"].values
        benchmark_metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 3)
        benchmark_metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 3)
        benchmark_metrics["f1_score"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 3)
        benchmark_metrics["ground_truth_count"] = int(y_true.sum())
        benchmark_metrics["detected_count"] = int(y_pred.sum())

    return df_all, benchmark_metrics


def create_anomaly_scatter_chart(df_anomalies: pd.DataFrame) -> go.Figure:
    """Create interactive time series chart highlighting detected anomalies."""
    fig = go.Figure()

    # Normal observations
    normal_df = df_anomalies[~df_anomalies["is_anomaly"]]
    anom_df = df_anomalies[df_anomalies["is_anomaly"]]

    fig.add_trace(
        go.Scatter(
            x=normal_df["date"],
            y=normal_df["temperature_c"],
            mode="lines+markers",
            name="Normal Observations",
            line=dict(color="#94a3b8", width=1.5),
            marker=dict(size=4, color="#64748b"),
        )
    )

    # Highlighted Anomalies
    if not anom_df.empty:
        fig.add_trace(
            go.Scatter(
                x=anom_df["date"],
                y=anom_df["temperature_c"],
                mode="markers",
                name=f"Anomalies ({len(anom_df)})",
                marker=dict(
                    size=11,
                    color="#ef4444",
                    symbol="diamond",
                    line=dict(width=2, color="#7f1d1d"),
                ),
                text=anom_df["anomaly_reason"],
                hovertemplate="<b>Date:</b> %{x}<br><b>Temp:</b> %{y}°C<br><b>Reason:</b> %{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Climate Anomaly Detection (Dual-Method Consensus)",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature (°C)"),
        hovermode="closest",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def save_anomaly_static_figures(
    df_anomalies: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> Path:
    """Save static anomaly plot for reporting."""
    if output_dir is None:
        base_dir = Path(__file__).resolve().parent.parent
        output_dir = base_dir / "outputs" / "figures"

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_anomalies["date"], df_anomalies["temperature_c"], color="#94a3b8", lw=1.2, label="Temperature")

    anom = df_anomalies[df_anomalies["is_anomaly"]]
    if not anom.empty:
        ax.scatter(
            anom["date"],
            anom["temperature_c"],
            color="#dc2626",
            s=55,
            edgecolors="#7f1d1d",
            label=f"Detected Anomalies (N={len(anom)})",
            zorder=5,
        )

    ax.set_title("Identified Temperature & Climate Anomalies", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "climate_anomalies_scatter.png", dpi=300)
    plt.close()

    return output_dir
